"""Integration test for the JKLU Compass RAG pipeline."""

from jklu_compass.rag.pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline(top_k=5)

    questions = [
        "What academic programs does JKLU offer?",
        "What is the official website of JKLU?",
        "What does JKLU's M.Des program focus on?",
        "What programs are available in engineering?",
        "What is the exact attendance requirement at JKLU?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        response = pipeline.ask(question)

        print("\nANSWER:\n")
        print(response.answer)

        print(f"\nRetrieved chunks: {response.retrieved_chunks}")

        print("\nSOURCES:\n")

        if not response.sources:
            print("No sources were retrieved.")
            continue

        for source in response.sources:
            print(f"[Source {source.number}] {source.title}")
            print(f"URL: {source.url or 'Unavailable'}")
            print(f"Chunk ID: {source.chunk_id}")
            print()


if __name__ == "__main__":
    main()
