"""Read-only diagnostic for dense, BM25 and RRF retrieval rankings."""

from pathlib import Path
import json

import chromadb

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.retrieval.retriever import DocumentRetriever
from jklu_compass.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"
BENCHMARK = ROOT / "data" / "evaluation" / "benchmark.json"

TOP_CANDIDATES = 60
FINAL_TOP_K = 6

TARGET_QUESTIONS = {
    "JKLU-BTECH-005",
    "JKLU-BTECH-007",
    "JKLU-BTECH-008",
    "JKLU-ADM-001",
    "JKLU-ADM-002",
    "JKLU-ADM-004",
    "JKLU-FAC-001",
    "JKLU-GEN-001",
    "JKLU-GEN-002",
    "JKLU-GEN-003",
}


def main():
    if not CHROMA_DIR.is_dir():
        raise RuntimeError("Existing ChromaDB directory not found.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)

    initial_count = collection.count()

    if initial_count == 0:
        raise RuntimeError("ChromaDB collection is empty.")

    store = ChromaVectorStore(
        persist_directory=CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    encoder = EmbeddingEncoder()

    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=store,
    )

    with BENCHMARK.open(encoding="utf-8") as file:
        benchmark = json.load(file)

    questions = [
        question for question in benchmark["questions"] if question["id"] in TARGET_QUESTIONS
    ]

    print(f"Collection: {DEFAULT_COLLECTION_NAME}")
    print(f"Initial records: {initial_count}")
    print(f"Questions inspected: {len(questions)}")
    print(f"Candidate depth: {TOP_CANDIDATES}")

    for question in questions:
        query = question["question"]
        relevant_ids = set(question["relevant_chunk_ids"])

        embedding = encoder.encode(query)

        # Dense candidates
        dense_records = store.search(
            query_embedding=embedding.tolist(),
            top_k=TOP_CANDIDATES,
        )

        dense_ids = [str(record["id"]) for record in dense_records]

        # BM25 candidates
        bm25_results = retriever.bm25_retriever.retrieve(
            query,
            top_k=TOP_CANDIDATES,
        )

        bm25_ids = [str(result.id) for result in bm25_results]

        # Reproduce RRF ranking
        scores = {}

        for rank, chunk_id in enumerate(dense_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (retriever.rrf_k + rank))

        for rank, chunk_id in enumerate(bm25_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (retriever.rrf_k + rank))

        rrf_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        # Actual final retrieval after source-diversity filtering
        final_results = retriever.retrieve(
            query,
            top_k=FINAL_TOP_K,
        )

        final_ids = [result.id for result in final_results]

        def rank_of(chunk_id, ranked_ids):
            try:
                return ranked_ids.index(chunk_id) + 1
            except ValueError:
                return None

        print("\n" + "=" * 90)
        print(f"{question['id']} | {question['category']}")
        print(f"QUESTION: {query}")
        print("=" * 90)

        for chunk_id in sorted(relevant_ids):
            print(f"\nRELEVANT CHUNK: {chunk_id}")
            print("Dense rank:", rank_of(chunk_id, dense_ids))
            print("BM25 rank:", rank_of(chunk_id, bm25_ids))
            print("RRF rank:", rank_of(chunk_id, rrf_ids))
            print("Final rank:", rank_of(chunk_id, final_ids))

        print("\nFINAL TOP SIX:")
        for rank, result in enumerate(final_results, start=1):
            status = "RELEVANT" if result.id in relevant_ids else "NOT LABELLED"
            print(
                f"{rank}. [{status}] {result.id} | "
                f"{result.title} | {result.metadata.get('section_title')}"
            )

    final_count = collection.count()

    print("\n" + "=" * 90)
    print("DATABASE VERIFICATION")
    print("Initial count:", initial_count)
    print("Final count:", final_count)

    if initial_count != final_count:
        raise RuntimeError("ChromaDB record count changed.")

    print("Read-only diagnostic completed.")


if __name__ == "__main__":
    main()
