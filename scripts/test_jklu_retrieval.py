"""
Test semantic retrieval against the indexed JKLU knowledge base.

Usage:
    python scripts/test_jklu_retrieval.py
    python scripts/test_jklu_retrieval.py --top-k 5
    python scripts/test_jklu_retrieval.py --query "What courses does JKLU offer?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jklu_compass.embeddings.encoder import EmbeddingEncoder  # noqa: E402
from jklu_compass.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "jklu_compass"

SAMPLE_QUERIES = [
    "What undergraduate programmes and courses does JKLU offer?",
    "What are the admission requirements and application process?",
    "What facilities and amenities are available on campus?",
    "Who are the faculty members and what departments do they belong to?",
    "What placement opportunities and career support does JKLU provide?",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test semantic retrieval for JKLU Compass.")

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to retrieve per query. Default: 3.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single custom query instead of the sample questions.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    if args.query is not None and not args.query.strip():
        parser.error("--query cannot be empty.")

    return args


def main() -> None:
    args = parse_arguments()

    if not CHROMA_DIR.exists():
        raise SystemExit(
            f"ChromaDB directory not found: {CHROMA_DIR}\nRun scripts/index_jklu_dataset.py first."
        )

    print("=" * 72)
    print("JKLU COMPASS — SEMANTIC RETRIEVAL TEST")
    print("=" * 72)

    store = ChromaVectorStore(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    vector_count = store.count()

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored vectors: {vector_count}")

    if vector_count == 0:
        raise SystemExit("The vector store is empty. Index the dataset before testing.")

    print("\nLoading embedding model...")

    encoder = EmbeddingEncoder()

    queries = [args.query] if args.query else SAMPLE_QUERIES

    successful_queries = 0

    for query_number, query in enumerate(queries, start=1):
        print("\n" + "=" * 72)
        print(f"QUERY {query_number}: {query}")
        print("=" * 72)

        # Convert the question into a vector.
        query_embedding = encoder.encode(query)

        # Retrieve the most similar indexed passages.
        results = store.search(
            query_embedding=query_embedding.tolist(),
            top_k=args.top_k,
        )

        if not results:
            print("No results returned.")
            continue

        successful_queries += 1

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            source = metadata.get("source", "Unknown source")
            file_name = metadata.get("file_name", "")
            chunk_index = metadata.get("chunk_index", "Unknown")
            distance = result["distance"]
            document = result["document"]

            print(f"\nResult #{rank}")
            print(f"Source: {source}")

            if file_name:
                print(f"File: {file_name}")

            print(f"Chunk index: {chunk_index}")
            print(f"Cosine distance: {distance:.4f}")
            print("-" * 72)

            preview = document[:900].strip()

            if len(document) > 900:
                preview += "..."

            print(preview)

    print("\n" + "=" * 72)
    print("RETRIEVAL TEST SUMMARY")
    print("=" * 72)
    print(f"Queries executed: {len(queries)}")
    print(f"Queries returning results: {successful_queries}")
    print(f"Stored vectors: {vector_count}")

    if successful_queries == len(queries):
        print("STATUS: Retrieval returned results for every query.")
    else:
        print("STATUS: Some queries did not return results.")

    print("\nNote: Results should be evaluated for relevance, not just")
    print("whether the vector database returned a passage.")


if __name__ == "__main__":
    main()
