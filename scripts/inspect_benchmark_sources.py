"""Inspect selected source documents for benchmark ground-truth preparation.

Read-only: does not modify ChromaDB.
"""

from collections import defaultdict
from pathlib import Path

from jklu_compass.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"

SOURCES = {
    "academics": [
        "https://jklu.edu.in/institute/engineering-technology",
        "https://jklu.edu.in/institute/design",
        "https://jklu.edu.in/uploads/course/pdf/Engineering-JKLU-New-Brochure-March2025-V9%20(2).pdf",
    ],
    "admissions": [
        "https://jklu.edu.in/admissions/fee-structure",
        "https://jklu.edu.in/admissions/scholarship-aid",
        "https://jklu.edu.in/admissions/international-admissions",
        "https://jklu.edu.in/uploads/pdf/university-prospectus.pdf",
    ],
    "placements": [
        "https://jklu.edu.in/placements",
        "https://jklu.edu.in/uploads/pdf/university-prospectus.pdf",
    ],
    "campus": [
        "https://jklu.edu.in/pdf/hostel-policy.pdf",
        "https://jklu.edu.in/campus-life/facilities/fitness-centre",
        "https://jklu.edu.in/campus-life/facilities/makerspace",
        "https://jklu.edu.in/campus-life/festivals",
    ],
    "faculty_and_research": [
        "https://jklu.edu.in/faculty/akshat-agrawal",
        "https://jklu.edu.in/faculty/sheetal-mundra",
        "https://jklu.edu.in/faculty/ashutosh-kumar",
        "https://jklu.edu.in/academics/centres",
    ],
    "general": [
        "https://jklu.edu.in/about-us/about-jklu",
        "https://jklu.edu.in/",
        "https://jklu.edu.in/siteassets/pdf/JKLU-Act.pdf",
    ],
}


def main() -> None:
    if not CHROMA_DIRECTORY.is_dir():
        raise RuntimeError(f"ChromaDB directory not found: {CHROMA_DIRECTORY}")

    store = ChromaVectorStore(
        persist_directory=CHROMA_DIRECTORY,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    initial_count = store.count()
    records = store.get_all_records(batch_size=500)

    if len(records) != initial_count:
        raise RuntimeError("Corpus read count does not match collection count.")

    by_source = defaultdict(list)

    for record in records:
        source_url = record["metadata"].get("source_url")
        if source_url:
            by_source[source_url].append(record)

    for category, urls in SOURCES.items():
        print(f"\n{'=' * 90}")
        print(f"CATEGORY: {category.upper()}")
        print("=" * 90)

        for url in urls:
            matching = by_source.get(url, [])

            print(f"\nSOURCE: {url}")
            print(f"CHUNKS: {len(matching)}")

            if not matching:
                print("No exact source_url match found.")
                continue

            matching.sort(key=lambda item: item["metadata"].get("chunk_index", 0))

            for record in matching:
                metadata = record["metadata"]

                print(f"\nChunk ID: {record['id']}")
                print(f"Chunk index: {metadata.get('chunk_index')}")
                print(f"Title: {metadata.get('title')}")
                print(f"Section: {metadata.get('section_title')}")
                print(f"Heading: {metadata.get('heading_path')}")
                print("Content:")
                print(record["document"][:1800])
                print("-" * 70)

    final_count = store.count()

    print("\n========== DATABASE VERIFICATION ==========")
    print(f"Initial count: {initial_count}")
    print(f"Final count:   {final_count}")

    if initial_count != final_count:
        raise RuntimeError("ChromaDB record count changed!")


if __name__ == "__main__":
    main()
