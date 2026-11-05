"""Inspect the existing JKLU Compass corpus for benchmark preparation.

This script only reads ChromaDB. It does not create, update, or delete records.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import chromadb

from jklu_compass.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "corpus_inventory.json"


def main() -> None:
    if not CHROMA_DIRECTORY.is_dir():
        raise RuntimeError(f"ChromaDB directory not found: {CHROMA_DIRECTORY}")

    # Verify the existing collection before opening the project wrapper.
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))

    try:
        collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(f"Existing collection '{DEFAULT_COLLECTION_NAME}' not found.") from exc

    initial_count = collection.count()

    if initial_count == 0:
        raise RuntimeError("Existing collection is empty.")

    store = ChromaVectorStore(
        persist_directory=CHROMA_DIRECTORY,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    records = store.get_all_records(batch_size=500)

    if len(records) != initial_count:
        raise RuntimeError(f"Record count mismatch: expected {initial_count}, read {len(records)}.")

    metadata_keys = Counter()
    source_counts = Counter()
    title_counts = Counter()
    category_counts = Counter()
    samples_by_source = defaultdict(list)

    for record in records:
        metadata = record["metadata"]
        metadata_keys.update(metadata.keys())

        source = (
            metadata.get("source_url") or metadata.get("source") or metadata.get("url") or "unknown"
        )

        title = metadata.get("title") or "Untitled"
        category = metadata.get("category") or metadata.get("document_type") or "uncategorized"

        source_counts[source] += 1
        title_counts[title] += 1
        category_counts[category] += 1

        if len(samples_by_source[source]) < 2:
            samples_by_source[source].append(
                {
                    "id": record["id"],
                    "title": title,
                    "section_title": metadata.get("section_title"),
                    "heading_path": metadata.get("heading_path"),
                    "document": record["document"][:700],
                    "metadata": metadata,
                }
            )

    inventory = {
        "collection": DEFAULT_COLLECTION_NAME,
        "record_count": initial_count,
        "metadata_keys": dict(metadata_keys.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "title_counts": dict(title_counts.most_common()),
        "samples_by_source": dict(samples_by_source),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    final_count = collection.count()

    if final_count != initial_count:
        raise RuntimeError("ChromaDB record count changed during inspection.")

    print("========== CORPUS INVENTORY ==========")
    print(f"Collection: {DEFAULT_COLLECTION_NAME}")
    print(f"Records read: {len(records)}")
    print(f"Unique sources: {len(source_counts)}")

    print("\nMetadata fields:")
    for key, count in metadata_keys.most_common():
        print(f"  {key}: {count}")

    print("\nDocument categories:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")

    print("\nTop document sources:")
    for source, count in source_counts.most_common(30):
        print(f"  [{count:4}] {source}")

    print("\nChromaDB verification:")
    print(f"Initial records: {initial_count}")
    print(f"Final records:   {final_count}")

    print(f"\nInventory saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
