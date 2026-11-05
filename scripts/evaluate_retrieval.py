"""Run read-only retrieval evaluation for JKLU Compass."""

from pathlib import Path

import chromadb

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.evaluation.evaluator import RetrievalEvaluator
from jklu_compass.retrieval.retriever import DocumentRetriever
from jklu_compass.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluation" / "benchmark.json"
RESULTS_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "results"


def build_read_only_retriever() -> tuple[DocumentRetriever, int]:
    """Connect to the existing collection without modifying its records."""

    if not CHROMA_DIRECTORY.is_dir():
        raise RuntimeError(f"Existing ChromaDB directory not found: {CHROMA_DIRECTORY}")

    # Verify the existing collection before constructing the vector store.
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))

    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Existing collection '{DEFAULT_COLLECTION_NAME}' not found. "
            "Refusing to create a new collection."
        ) from exc

    initial_count = collection.count()

    if initial_count == 0:
        raise RuntimeError("Existing ChromaDB collection is empty.")

    print(f"Connected to existing collection: {DEFAULT_COLLECTION_NAME}")
    print(f"Initial record count: {initial_count}")

    # This constructor uses get_or_create_collection internally.
    # The preflight above ensures the expected collection already exists.
    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_DIRECTORY,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    encoder = EmbeddingEncoder()

    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
    )

    return retriever, initial_count


def main() -> None:
    """Run benchmark evaluation and verify collection count afterward."""

    retriever, initial_count = build_read_only_retriever()

    evaluator = RetrievalEvaluator(
        retriever=retriever,
        benchmark_path=BENCHMARK_PATH,
        results_dir=RESULTS_DIRECTORY,
        top_k=6,
    )

    print("\nStarting retrieval evaluation...\n")

    report = evaluator.evaluate(save=True)

    final_count = retriever.vector_store.count()

    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Questions evaluated: {report['total_questions']}")
    print(f"Top K: {report['top_k']}")

    print("\nAggregate metrics:")

    for metric, value in report["aggregate_metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nCollection verification:")
    print(f"Initial records: {initial_count}")
    print(f"Final records:   {final_count}")

    if initial_count != final_count:
        raise RuntimeError(
            "Collection count changed during evaluation. Investigate before continuing."
        )

    print("\nReport saved:")
    print(report["report_path"])
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
