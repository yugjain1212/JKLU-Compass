"""Inspect source metadata stored in the JKLU Compass ChromaDB index."""

from collections import Counter

from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    store = ChromaVectorStore()

    print(f"Total indexed chunks: {store.count()}")

    # Read stored records directly; no embedding or model loading is needed.
    records = store.collection.get(
        include=["metadatas"],
        limit=store.count(),
    )

    metadatas = records.get("metadatas") or []
    title_values = Counter()
    bad_titles = []

    for metadata in metadatas:
        metadata = metadata or {}
        title = metadata.get("title")
        file_name = metadata.get("file_name")
        source = metadata.get("source")

        title_values[str(title)] += 1

        if (
            not title
            or str(title).strip().lower() in {"undefined", "none", "null"}
        ):
            bad_titles.append(
                {
                    "title": title,
                    "file_name": file_name,
                    "source": source,
                    "source_url": metadata.get("source_url"),
                }
            )

    print("\nMost common stored title values:")
    for title, count in title_values.most_common(15):
        print(f"{count:5} | {title}")

    print(f"\nChunks with missing or invalid titles: {len(bad_titles)}")
    for item in bad_titles[:10]:
        print("\n---")
        for key, value in item.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
