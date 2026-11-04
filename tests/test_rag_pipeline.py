"""Tests for the JKLU Compass RAG pipeline."""

from unittest.mock import MagicMock

import pytest

from jklu_compass.rag.pipeline import RAGPipeline
from jklu_compass.retrieval.retriever import RetrievalResult


@pytest.fixture
def retriever() -> MagicMock:
    """Provide a mocked document retriever."""
    return MagicMock()


@pytest.fixture
def llm() -> MagicMock:
    """Provide a mocked local language model."""
    mock = MagicMock()
    mock.generate.return_value = "JKLU offers several undergraduate programmes. [Source 1]"
    return mock


@pytest.fixture
def pipeline(
    retriever: MagicMock,
    llm: MagicMock,
) -> RAGPipeline:
    """Create a RAG pipeline with mocked dependencies."""
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=5,
    )


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Provide representative retrieved document chunks."""
    return [
        RetrievalResult(
            id="chunk-001",
            content="JKLU offers B.Tech programmes in engineering.",
            title="Engineering Brochure",
            source_url="https://jklu.edu.in/engineering.pdf",
            source="/data/engineering.md",
            distance=0.12,
            metadata={"section_title": "B.Tech Programmes"},
        ),
        RetrievalResult(
            id="chunk-002",
            content="JKLU offers undergraduate design programmes.",
            title="Design Brochure",
            source_url="https://jklu.edu.in/design.pdf",
            source="/data/design.md",
            distance=0.18,
            metadata={"section_title": "Design Programmes"},
        ),
    ]


def test_ask_retrieves_documents_and_generates_answer(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Retrieve context and generate a grounded answer."""
    retriever.retrieve.return_value = sample_results

    response = pipeline.ask("What programmes does JKLU offer?")

    retriever.retrieve.assert_called_once_with(
        "What programmes does JKLU offer?",
        top_k=5,
    )
    llm.generate.assert_called_once()

    assert response.question == "What programmes does JKLU offer?"
    assert response.answer == ("JKLU offers several undergraduate programmes. [Source 1]")
    assert response.retrieved_chunks == 2
    assert len(response.sources) == 2


def test_ask_returns_correct_source_citations(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Preserve citation numbers, URLs, titles, and chunk IDs."""
    retriever.retrieve.return_value = sample_results

    response = pipeline.ask("What programmes does JKLU offer?")

    assert response.sources[0].number == 1
    assert response.sources[0].title == "Engineering Brochure"
    assert response.sources[0].url == "https://jklu.edu.in/engineering.pdf"
    assert response.sources[0].chunk_id == "chunk-001"

    assert response.sources[1].number == 2
    assert response.sources[1].title == "Design Brochure"
    assert response.sources[1].chunk_id == "chunk-002"


def test_ask_does_not_call_llm_when_retrieval_is_empty(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """Return a fallback response when no documents are retrieved."""
    retriever.retrieve.return_value = []

    response = pipeline.ask("What are the JKLU fees?")

    llm.generate.assert_not_called()

    assert response.sources == ()
    assert response.retrieved_chunks == 0
    assert "couldn't find usable information" in response.answer


def test_ask_filters_empty_chunks_from_context(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Exclude empty chunks and maintain consecutive citation numbers."""
    empty_result = RetrievalResult(
        id="empty-chunk",
        content="   ",
        title="Empty document",
        source_url=None,
        source="/data/empty.md",
        distance=0.1,
        metadata={},
    )

    retriever.retrieve.return_value = [
        empty_result,
        sample_results[0],
    ]

    response = pipeline.ask("What does JKLU offer?")

    assert len(response.sources) == 1
    assert response.sources[0].number == 1
    assert response.sources[0].chunk_id == "chunk-001"

    prompt = llm.generate.call_args.args[0]
    assert "Empty document" not in prompt
    assert "[Source 1]" in prompt


def test_ask_includes_grounding_instructions(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Ensure the prompt contains evidence and citation rules."""
    retriever.retrieve.return_value = sample_results

    pipeline.ask("What programmes does JKLU offer?")

    prompt = llm.generate.call_args.args[0]

    assert "Never invent university facts" in prompt
    assert "reference passages as evidence" in prompt
    assert "Do not invent citations or URLs" in prompt
    assert "Engineering Brochure" in prompt
    assert "Design Brochure" in prompt


def test_ask_uses_low_temperature(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Use a low temperature for factual university responses."""
    retriever.retrieve.return_value = sample_results

    pipeline.ask("What programmes does JKLU offer?")

    llm.generate.assert_called_once()
    assert llm.generate.call_args.kwargs["temperature"] == 0.1


def test_ask_rejects_empty_question(
    pipeline: RAGPipeline,
) -> None:
    """Reject invalid questions before retrieval."""
    with pytest.raises(ValueError, match="(?i)question"):
        pipeline.ask("   ")


def test_ask_normalizes_question_whitespace(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Strip surrounding whitespace before retrieval."""
    retriever.retrieve.return_value = sample_results

    response = pipeline.ask("  What programmes does JKLU offer?  ")

    assert response.question == "What programmes does JKLU offer?"
    retriever.retrieve.assert_called_once_with(
        "What programmes does JKLU offer?",
        top_k=5,
    )


def test_ask_ignores_invalid_source_titles(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Replace invalid document titles with a safe fallback."""
    invalid_title_result = RetrievalResult(
        id="chunk-003",
        content="Some relevant JKLU information.",
        title="undefined",
        source_url=None,
        source="/data/unknown.md",
        distance=0.2,
        metadata={},
    )

    retriever.retrieve.return_value = [
        invalid_title_result,
        sample_results[0],
    ]

    response = pipeline.ask("Tell me about JKLU.")

    assert response.sources[0].title == "Untitled JKLU document"
