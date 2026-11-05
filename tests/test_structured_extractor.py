"""Tests for structured document extraction."""

from jklu_compass.rag.structured_extractor import StructuredExtractor
from jklu_compass.retrieval.retriever import RetrievalResult


def make_result(
    name: str,
    chunk_id: str,
    heading_path: list[str],
) -> RetrievalResult:
    return RetrievalResult(
        id=chunk_id,
        content=f"Information about {name}.",
        title="Engineering Brochure",
        source_url="https://jklu.edu.in/engineering.pdf",
        source="/data/engineering.md",
        distance=0.1,
        metadata={
            "section_title": name,
            "heading_path": heading_path,
            "heading_level": 5,
        },
    )


def test_extracts_btech_programmes_in_order() -> None:
    results = [
        make_result(
            "Computer Science and Engineering",
            "chunk-1",
            [
                "# TECHNOLOGY",
                "### Programmes",
                "#### B.TECH BACHELOR OF TECHNOLOGY",
                "##### Computer Science and Engineering",
            ],
        ),
        make_result(
            "Computer Science and Engineering (Artificial Intelligence)",
            "chunk-2",
            [
                "# TECHNOLOGY",
                "#### B.TECH BACHELOR OF TECHNOLOGY",
                "##### Computer Science and Engineering",
                "##### Computer Science and Engineering (Artificial Intelligence)",
            ],
        ),
    ]

    items = StructuredExtractor.extract_btech_programmes(results)

    assert len(items) == 2
    assert items[0].name == "Computer Science and Engineering"
    assert items[1].name == ("Computer Science and Engineering (Artificial Intelligence)")


def test_ignores_non_btech_records() -> None:
    result = make_result(
        "Bachelor of Design",
        "chunk-design",
        [
            "# DESIGN",
            "#### B.DES PROGRAMMES",
            "##### Bachelor of Design",
        ],
    )

    assert StructuredExtractor.extract_btech_programmes([result]) == []


def test_deduplicates_programme_names() -> None:
    first = make_result(
        "Computer Science and Engineering",
        "chunk-1",
        ["#### B.TECH BACHELOR OF TECHNOLOGY"],
    )
    second = make_result(
        "Computer Science and Engineering",
        "chunk-2",
        ["#### B.TECH BACHELOR OF TECHNOLOGY"],
    )

    items = StructuredExtractor.extract_btech_programmes([first, second])

    assert len(items) == 1
    assert items[0].chunk_id == "chunk-1"


def test_preserves_source_information() -> None:
    result = make_result(
        "Computer Science and Engineering",
        "chunk-1",
        ["#### B.TECH BACHELOR OF TECHNOLOGY"],
    )

    items = StructuredExtractor.extract_btech_programmes([result])

    assert items[0].chunk_id == "chunk-1"
    assert items[0].source_url == ("https://jklu.edu.in/engineering.pdf")
    assert items[0].source_title == "Engineering Brochure"
