"""Inspect indexed chunks containing academic program information."""

from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    store = ChromaVectorStore()
    records = store.collection.get(
        include=["documents", "metadatas"],
        limit=store.count(),
    )

    keywords = (
        "academic programmes",
        "academic programs",
        "undergraduate",
        "postgraduate",
        "doctoral",
        "bachelor of",
        "master of",
        "ph.d",
        "b.tech",
        "m.tech",
        "b.des",
        "m.des",
        "bba",
        "mba",
    )

    matches = []

    for document, metadata in zip(
        records["documents"],
        records["metadatas"],
        strict=True,
    ):
        content = document or ""
        lowered = content.lower()

        matched = [keyword for keyword in keywords if keyword in lowered]

        if len(matched) >= 2 or (
            "academic programmes" in lowered or "academic programs" in lowered
        ):
            matches.append(
                {
                    "title": (metadata or {}).get("title", "Unknown"),
                    "source": (metadata or {}).get("source", "Unknown"),
                    "matched": matched,
                    "content": content,
                }
            )

    print(f"Matching chunks: {len(matches)}")

    for index, item in enumerate(matches[:30], start=1):
        print(f"\n{'=' * 80}")
        print(f"RESULT {index}")
        print(f"Title: {item['title']}")
        print(f"Source: {item['source']}")
        print(f"Keywords: {', '.join(item['matched'])}")
        print("\nContent:")
        print(item["content"][:1800])


if __name__ == "__main__":
    main()
