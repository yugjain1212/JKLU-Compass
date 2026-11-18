"""FastAPI application for JKLU Compass."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import chromadb
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from jklu_compass.api.clarification_state import ClarificationState
from jklu_compass.api.schemas import (
    ClarificationOptionResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    StructuredAnswerResponse,
    StructuredProgrammeResponse,
)
from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.generation.local_llm import LocalLLM, LocalLLMError
from jklu_compass.rag.pipeline import RAGPipeline
from jklu_compass.rag.query_clarifier import QueryClarifier
from jklu_compass.rag.query_reconstructor import QueryReconstructor
from jklu_compass.retrieval.retriever import DocumentRetriever
from jklu_compass.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"


def build_rag_pipeline() -> RAGPipeline:
    """Build the RAG pipeline using the existing persistent ChromaDB."""

    # Never silently create a new database or collection.
    if not CHROMA_DIRECTORY.is_dir():
        raise RuntimeError(f"ChromaDB directory does not exist: {CHROMA_DIRECTORY}")

    # Verify that the expected collection already exists.
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))

    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Existing ChromaDB collection '{DEFAULT_COLLECTION_NAME}' could not be found."
        ) from exc

    record_count = collection.count()

    if record_count == 0:
        raise RuntimeError("The existing ChromaDB collection is empty. Refusing to start the API.")

    # Connect to the verified existing collection.
    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_DIRECTORY,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    # Load the embedding model once.
    encoder = EmbeddingEncoder()

    # DocumentRetriever builds the in-memory BM25 index from existing records.
    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
    )

    # LocalLLM connects to Ollama when generating an answer.
    llm = LocalLLM()

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=8,
    )


def create_app(
    pipeline_factory: Callable[[], RAGPipeline] = build_rag_pipeline,
    clarification_ttl_seconds: int = 600,
) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        """Initialize the RAG pipeline once during application startup."""
        app.state.rag_pipeline = pipeline_factory()
        llm = getattr(app.state.rag_pipeline, "llm", None)
        app.state.query_clarifier = QueryClarifier(llm) if llm is not None else None
        yield
        app.state.rag_pipeline = None
        app.state.query_clarifier = None

    app = FastAPI(
        title="JKLU Compass API",
        description=(
            "Retrieval-augmented university knowledge assistant for JK Lakshmipat University."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.clarification_state = ClarificationState(clarification_ttl_seconds)

    @app.exception_handler(LocalLLMError)
    async def handle_llm_error(
        request: Request,
        exc: LocalLLMError,
    ) -> JSONResponse:
        """Return a service-unavailable response when Ollama fails."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": (
                    "The local language model is currently unavailable. "
                    "Check that Ollama is running and the configured model is available."
                )
            },
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
    )
    def health(request: Request) -> HealthResponse:
        """Check whether the API and RAG pipeline are initialized."""
        pipeline = getattr(request.app.state, "rag_pipeline", None)

        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline is not initialized.",
            )

        return HealthResponse(
            status="ok",
            pipeline="ready",
        )

    @app.post(
        "/api/v1/query",
        response_model=QueryResponse,
        responses={
            503: {"description": "Local language model unavailable."},
        },
        tags=["RAG"],
    )
    def query(
        payload: QueryRequest,
        request: Request,
    ) -> QueryResponse:
        """Clarify when useful, then answer through the existing RAG pipeline."""
        pipeline: RAGPipeline = request.app.state.rag_pipeline
        conversation_id = payload.conversation_id or str(uuid4())
        state: ClarificationState = request.app.state.clarification_state
        display_question = payload.question or ""

        if payload.clarification_id:
            try:
                pending, selected = state.consume(
                    conversation_id,
                    payload.clarification_id,
                    payload.option_id or "",
                    payload.custom_answer,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid clarification option.",
                ) from exc
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=410,
                    detail="This clarification expired. Please ask your question again.",
                ) from exc
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="This clarification was already submitted.",
                ) from exc
            except LookupError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="This clarification is stale. Please ask your question again.",
                ) from exc

            question = QueryReconstructor.reconstruct(
                pending.question,
                selected.question,
            )
            display_question = f"{pending.question} ({selected.question})"
            result = pipeline.ask(question, history=pending.history)
        else:
            question = payload.question or ""
            history = tuple((turn.role, turn.content) for turn in payload.history)
            clarifier: QueryClarifier | None = request.app.state.query_clarifier
            decision = (
                clarifier.clarify(question, history, payload.clarification_mode)
                if clarifier is not None
                else None
            )
            if decision is not None and decision.required:
                clarification_id = str(uuid4())
                pending = state.create(
                    conversation_id,
                    clarification_id,
                    question,
                    history,
                    decision.options,
                )
                return QueryResponse(
                    status="clarification_required",
                    question=question,
                    conversation_id=conversation_id,
                    clarification_id=pending.clarification_id,
                    clarification_message=decision.message,
                    options=[
                        ClarificationOptionResponse(id=option.id, question=option.question)
                        for option in decision.options
                    ],
                )

            result = pipeline.ask(question, history=history)

        structured_data = None

        if result.structured_data is not None:
            structured_data = StructuredAnswerResponse(
                answer_type=result.structured_data.answer_type,
                programmes=[
                    StructuredProgrammeResponse(
                        name=programme.name,
                        source_number=programme.source_number,
                    )
                    for programme in result.structured_data.programmes
                ],
            )

        return QueryResponse(
            status="answer",
            question=display_question,
            answer=result.answer,
            sources=[
                SourceResponse(
                    number=source.number,
                    title=source.title,
                    url=source.url,
                    chunk_id=source.chunk_id,
                )
                for source in result.sources
            ],
            retrieved_chunks=result.retrieved_chunks,
            structured_data=structured_data,
            conversation_id=conversation_id,
        )

    return app


app = create_app()
