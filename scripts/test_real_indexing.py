"""Smoke test for real embeddings and ChromaDB indexing."""

from pathlib import Path
from tempfile import TemporaryDirectory

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.ingestion.indexer import DocumentIndexer
from jklu_compass.ingestion.pipeline import IngestionPipeline
from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    """Index a tiny sample and verify semantic retrieval."""
    with TemporaryDirectory(prefix="jklu_compass_test_") as temp_dir:
        root = Path(temp_dir)
        source_dir = root / "sample_documents"
        source_dir.mkdir()

        (source_dir / "admissions.md").write_text(
            "# Admissions\n\n"
            "JKLU students can find information about admission "
            "requirements and the application process here.",
            encoding="utf-8",
        )

        (source_dir / "campus.md").write_text(
            "# Campus\n\n"
            "The university campus includes facilities for "
            "student activities and academic learning.",
            encoding="utf-8",
        )

        encoder = EmbeddingEncoder()
        store = ChromaVectorStore(
            persist_directory=root / "chroma",
            collection_name="smoke_test",
        )

        indexer = DocumentIndexer(
            encoder=encoder,
            vector_store=store,
            ingestion_pipeline=IngestionPipeline(),
            batch_size=2,
        )

        report = indexer.index_directory(source_dir)

        print("\nIndexing report:")
        print(f"Documents loaded: {report.documents_loaded}")
        print(f"Chunks created:   {report.chunks_created}")
        print(f"Chunks indexed:   {report.chunks_indexed}")
        print(f"Stored records:   {report.vector_store_count}")

        query = "How can students learn about admission requirements?"
        query_embedding = encoder.encode(query)
        results = store.search(query_embedding.tolist(), top_k=2)

        print("\nRetrieval results:")
        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. {result['metadata'].get('source', 'Unknown source')}")
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Text: {result['document']}")

        assert report.documents_loaded == 2
        assert report.chunks_indexed == report.chunks_created
        assert store.count() == report.chunks_indexed
        assert results, "Retrieval returned no results."

        print("\nSmoke test passed.")


if __name__ == "__main__":
    main()