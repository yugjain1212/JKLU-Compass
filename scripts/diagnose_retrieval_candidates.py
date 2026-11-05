"""Read-only, production-aligned retrieval diagnostics for the benchmark."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.evaluation.dataset import BenchmarkQuestion, load_benchmark
from jklu_compass.retrieval.retriever import DocumentRetriever, RetrievalResult
from jklu_compass.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"
BENCHMARK = ROOT / "data" / "evaluation" / "benchmark.json"
DIAGNOSTIC_DIR = ROOT / "data" / "evaluation" / "diagnostics"

EXPECTED_RECORD_COUNT = 5716
FINAL_TOP_K = 6
PRODUCTION_CANDIDATE_MULTIPLIER = 3
EXPLORATORY_CANDIDATE_DEPTH = 60


def serialize_result(result: RetrievalResult) -> dict[str, Any]:
    """Return all stored evidence for a retrieved chunk."""
    return {
        "id": result.id,
        "content": result.content,
        "title": result.title,
        "source_url": result.source_url,
        "source": result.source,
        "distance": result.distance,
        "metadata": result.metadata,
    }


def diagnose_question(
    retriever: DocumentRetriever,
    question: BenchmarkQuestion,
    mode: str,
    top_k: int = FINAL_TOP_K,
) -> dict[str, Any]:
    """Run the shared production retrieval path and serialize its trace."""
    trace: dict[str, Any] = {}
    results = retriever.retrieve(question.question, top_k=top_k, trace=trace)
    final_ids = [result.id for result in results]
    final_ranks = {result.id: rank for rank, result in enumerate(results, start=1)}
    final_by_id = {result.id: result for result in results}
    ranked_candidates = trace["ranked_candidates"]
    candidate_by_id = {candidate["id"]: candidate for candidate in ranked_candidates}

    corpus_by_id = {
        str(record["id"]): record
        for record in (retriever.bm25_retriever.records if retriever.bm25_retriever else [])
    }

    expected: list[dict[str, Any]] = []
    for chunk_id in question.relevant_chunk_ids:
        candidate = candidate_by_id.get(chunk_id)
        record = corpus_by_id.get(chunk_id)

        if candidate is not None:
            result = candidate["result"]
            selection_reason = candidate.get("selection_reason")
        elif chunk_id in final_by_id:
            result = final_by_id[chunk_id]
            selection_reason = "selected_structured_programme"
        elif chunk_id in trace.get("missing_record_ids", []):
            result = retriever._build_result(record, distance=1.0)[0] if record else None
            selection_reason = "missing_candidate_record"
        else:
            result = retriever._build_result(record, distance=1.0)[0] if record else None
            selection_reason = "outside_candidate_lists"

        expected.append(
            {
                "id": chunk_id,
                "dense_rank": candidate.get("dense_rank") if candidate else None,
                "bm25_rank": candidate.get("bm25_rank") if candidate else None,
                "rrf_rank": candidate.get("rank") if candidate else None,
                "final_rank": final_ranks.get(chunk_id),
                "selection_reason": selection_reason,
                "chunk": serialize_result(result) if result else None,
            }
        )

    candidates = [
        {
            "rank": candidate["rank"],
            "id": candidate["id"],
            "dense_rank": candidate.get("dense_rank"),
            "bm25_rank": candidate.get("bm25_rank"),
            "rrf_score": candidate.get("rrf_score"),
            "source_key": candidate["source_key"],
            "selection_reason": candidate.get("selection_reason"),
            "chunk": serialize_result(candidate["result"]),
        }
        for candidate in ranked_candidates
    ]

    selection_reasons = trace.get("selection_reasons", {})
    final_results = [
        {
            "rank": rank,
            "selection_reason": selection_reasons.get(result.id, "structured_programme"),
            "chunk": serialize_result(result),
        }
        for rank, result in enumerate(results, start=1)
    ]

    return {
        "mode": mode,
        "candidate_depth": trace["candidate_k"],
        "selection_mode": trace["selection_mode"],
        "question_id": question.id,
        "question": question.question,
        "category": question.category,
        "expected_chunk_ids": list(question.relevant_chunk_ids),
        "dense_candidates": trace.get("dense_candidates", []),
        "bm25_candidates": trace.get("bm25_candidates", []),
        "missing_record_ids": trace.get("missing_record_ids", []),
        "expected_chunks": expected,
        "ranked_candidates": candidates,
        "retrieved_chunk_ids": final_ids,
        "final_results": final_results,
    }


def main() -> None:
    """Run exact-depth and exploratory diagnostics without changing ChromaDB."""
    if not CHROMA_DIR.is_dir():
        raise RuntimeError(f"Existing ChromaDB directory not found: {CHROMA_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Existing collection '{DEFAULT_COLLECTION_NAME}' not found; refusing to create it."
        ) from exc

    initial_count = collection.count()
    if initial_count != EXPECTED_RECORD_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORD_COUNT} records before diagnostics, found {initial_count}."
        )

    report: dict[str, Any] | None = None
    try:
        store = ChromaVectorStore(
            persist_directory=CHROMA_DIR,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
        encoder = EmbeddingEncoder()
        production_retriever = DocumentRetriever(
            encoder=encoder,
            vector_store=store,
            candidate_multiplier=PRODUCTION_CANDIDATE_MULTIPLIER,
        )
        if EXPLORATORY_CANDIDATE_DEPTH % FINAL_TOP_K:
            raise ValueError("Exploratory candidate depth must be divisible by top_k.")

        exploratory_retriever = DocumentRetriever(
            encoder=encoder,
            vector_store=store,
            candidate_multiplier=EXPLORATORY_CANDIDATE_DEPTH // FINAL_TOP_K,
            bm25_retriever=production_retriever.bm25_retriever,
        )
        questions = load_benchmark(BENCHMARK)

        question_reports = []
        for question in questions:
            production = diagnose_question(
                production_retriever,
                question,
                mode="production_equivalent",
            )
            exploratory = diagnose_question(
                exploratory_retriever,
                question,
                mode="exploratory_deeper_candidates",
            )
            question_reports.append(
                {
                    "question_id": question.id,
                    "production_equivalent": production,
                    "exploratory": exploratory,
                }
            )

            print(f"\n{question.id} | {question.category} | {question.question}")
            for item in production["expected_chunks"]:
                print(
                    f"  expected {item['id']}: dense={item['dense_rank']} "
                    f"bm25={item['bm25_rank']} rrf={item['rrf_rank']} "
                    f"final={item['final_rank']} "
                    f"reason={item['selection_reason']}"
                )
            print("  production final IDs: " + ", ".join(production["retrieved_chunk_ids"]))
            print("  exploratory final IDs: " + ", ".join(exploratory["retrieved_chunk_ids"]))

        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "collection": DEFAULT_COLLECTION_NAME,
            "initial_record_count": initial_count,
            "top_k": FINAL_TOP_K,
            "production_candidate_multiplier": PRODUCTION_CANDIDATE_MULTIPLIER,
            "production_candidate_depth": (FINAL_TOP_K * PRODUCTION_CANDIDATE_MULTIPLIER),
            "exploratory_candidate_depth": EXPLORATORY_CANDIDATE_DEPTH,
            "total_questions": len(questions),
            "questions": question_reports,
        }
    finally:
        final_count = collection.count()
        print("\n========== READ-ONLY DATABASE VERIFICATION ==========")
        print(f"Initial records: {initial_count}")
        print(f"Final records:   {final_count}")
        if final_count != initial_count:
            raise RuntimeError("ChromaDB record count changed during diagnostics.")

    if report is None:
        raise RuntimeError("Diagnostic report was not produced.")

    report["final_record_count"] = final_count
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = DIAGNOSTIC_DIR / f"retrieval_diagnostic_{timestamp}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Diagnostic report saved: {report_path}")


if __name__ == "__main__":
    main()
