"""Inspect retrieved chunks to evaluate JKLU Compass retrieval quality."""

from jklu_compass.retrieval.retriever import DocumentRetriever


QUESTIONS = [
    "What academic programs does JKLU offer?",
    "What does JKLU's M.Des program focus on?",
    "What is the exact attendance requirement at JKLU?",
]


def main() -> None:
    retriever = DocumentRetriever()

    for question in QUESTIONS:
        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")
        print("=" * 90)

        results = retriever.retrieve(question, top_k=5)

        if not results:
            print("No results returned.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n--- RESULT {rank} ---")
            print(f"Chunk ID: {result.id}")
            print(f"Title: {result.title}")
            print(f"Distance: {result.distance:.4f}")
            print(f"URL: {result.source_url or 'Unavailable'}")
            print("\nCONTENT:")
            print(result.content[:1800])

            if len(result.content) > 1800:
                print("\n[Content truncated for inspection]")


if __name__ == "__main__":
    main()