from jklu_compass.retrieval.retriever import DocumentRetriever

queries = [
    "What undergraduate programmes does JKLU offer?",
    "What are the research interests of Dr. Akshat Agrawal?",
    "How can students contact the admissions office?",
    "What placement support does JKLU provide?",
]

retriever = DocumentRetriever()

for query in queries:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    results = retriever.retrieve(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Title: {result.title}")
        print(f"Distance: {result.distance:.4f}")
        print(f"URL: {result.source_url}")
        print(f"Chunk ID: {result.id}")
        print(f"Content:\n{result.content[:1200]}")
        print("-" * 80)
