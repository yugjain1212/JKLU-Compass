"""Smoke test for real embeddings and ChromaDB indexing."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# ---------------------------------------------------------
# 1. Configure the project import path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# noqa: E402
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------
# 2. Import project modules
# ---------------------------------------------------------
from jklu_compass.embeddings.encoder import EmbeddingEncoder  # noqa: E402
from jklu_compass.ingestion.indexer import DocumentIndexer  # noqa: E402
from jklu_compass.ingestion.pipeline import IngestionPipeline  # noqa: E402
from jklu_compass.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402

# ---------------------------------------------------------
# 3. Main smoke-test function
# ---------------------------------------------------------


def main() -> None:
    """Test document ingestion, embedding, indexing, and retrieval."""

    with TemporaryDirectory(prefix="jklu_compass_test_") as temp_dir:
        root = Path(temp_dir)
        source_dir = root / "sample_documents"
        source_dir.mkdir()

        # -------------------------------------------------
        # 3.1 Create sample documents
        # -------------------------------------------------

        admissions_file = source_dir / "admissions.md"
        admissions_file.write_text(
            "# Admissions\n\n"
            "JKLU students can find information about admission "
            "requirements and the application process here.",
            encoding="utf-8",
        )

        campus_file = source_dir / "campus.md"
        campus_file.write_text(
            "# Campus\n\n"
            "The university campus includes facilities for "
            "student activities and academic learning.",
            encoding="utf-8",
        )

        print("=" * 60)
        print("JKLU COMPASS - REAL INDEXING SMOKE TEST")
        print("=" * 60)

        print("\nSample documents created:")
        print(f"1. {admissions_file.name}")
        print(f"2. {campus_file.name}")

        # -------------------------------------------------
        # 3.2 Initialize the embedding encoder
        # -------------------------------------------------

        print("\nLoading embedding model...")

        encoder = EmbeddingEncoder()

        print("Embedding model loaded successfully.")

        # -------------------------------------------------
        # 3.3 Initialize the ChromaDB vector store
        # -------------------------------------------------

        chroma_directory = root / "chroma"

        store = ChromaVectorStore(
            persist_directory=chroma_directory,
            collection_name="smoke_test",
        )

        print(f"ChromaDB directory: {chroma_directory}")

        # -------------------------------------------------
        # 3.4 Initialize the ingestion and indexing pipeline
        # -------------------------------------------------

        pipeline = IngestionPipeline()

        indexer = DocumentIndexer(
            encoder=encoder,
            vector_store=store,
            ingestion_pipeline=pipeline,
            batch_size=2,
        )

        # -------------------------------------------------
        # 3.5 Ingest and index the documents
        # -------------------------------------------------

        print("\nIndexing documents...")

        report = indexer.index_directory(source_dir)

        print("\n" + "=" * 60)
        print("INDEXING REPORT")
        print("=" * 60)

        print(f"Documents loaded: {report.documents_loaded}")
        print(f"Chunks created:   {report.chunks_created}")
        print(f"Chunks indexed:   {report.chunks_indexed}")
        print(f"Stored records:   {report.vector_store_count}")

        # -------------------------------------------------
        # 3.6 Generate an embedding for a search query
        # -------------------------------------------------

        query = "How can students learn about admission requirements?"

        print("\n" + "=" * 60)
        print("SEMANTIC RETRIEVAL")
        print("=" * 60)

        print(f"\nQuery: {query}")

        query_embedding = encoder.encode(query)

        # -------------------------------------------------
        # 3.7 Search the ChromaDB collection
        # -------------------------------------------------

        results = store.search(
            query_embedding.tolist(),
            top_k=2,
        )

        print(f"\nResults returned: {len(results)}")

        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            distance = result["distance"]
            document = result["document"]

            print(f"\nResult {rank}")
            print("-" * 40)
            print(f"Source:   {source}")
            print(f"Distance: {distance:.4f}")
            print(f"Text:     {document}")

        # -------------------------------------------------
        # 3.8 Validate the smoke-test results
        # -------------------------------------------------

        assert report.documents_loaded == 2, f"Expected 2 documents, got {report.documents_loaded}"

        assert report.chunks_created > 0, "The ingestion pipeline did not create any chunks."

        assert report.chunks_indexed == report.chunks_created, (
            f"Created {report.chunks_created} chunks but indexed {report.chunks_indexed}."
        )

        assert store.count() == report.chunks_indexed, (
            f"Expected {report.chunks_indexed} stored records, got {store.count()}."
        )

        assert results, "Semantic retrieval returned no results."

        assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}."

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)

        print("Document loading:       PASS")
        print("Text chunking:          PASS")
        print("Embedding generation:   PASS")
        print("ChromaDB indexing:      PASS")
        print("Semantic retrieval:     PASS")
        print("Result validation:      PASS")


# ---------------------------------------------------------
# 4. Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
