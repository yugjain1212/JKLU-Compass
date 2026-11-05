from jklu_compass.rag.pipeline import (
    RAGPipeline,
    SourceCitation,
    StructuredAnswer,
    StructuredProgramme,
)
from jklu_compass.rag.structured_extractor import StructuredItem


def test_builds_structured_btech_programme_response():
    items = [
        StructuredItem(
            name="Computer Science and Engineering",
            chunk_id="chunk-001",
            source_url="https://example.com/btech.pdf",
            source_title="Engineering Brochure",
        ),
        StructuredItem(
            name="Computer Science and Engineering (Artificial Intelligence)",
            chunk_id="chunk-002",
            source_url="https://example.com/btech.pdf",
            source_title="Engineering Brochure",
        ),
    ]

    citations = [
        SourceCitation(
            number=1,
            title="Engineering Brochure",
            url="https://example.com/btech.pdf",
            chunk_id="chunk-001",
        ),
        SourceCitation(
            number=2,
            title="Engineering Brochure",
            url="https://example.com/btech.pdf",
            chunk_id="chunk-002",
        ),
    ]

    structured = RAGPipeline._build_structured_answer(
        items,
        citations,
    )

    assert isinstance(structured, StructuredAnswer)
    assert structured.answer_type == "btech_programme_list"

    assert structured.programmes == (
        StructuredProgramme(
            name="Computer Science and Engineering",
            source_number=1,
        ),
        StructuredProgramme(
            name="Computer Science and Engineering (Artificial Intelligence)",
            source_number=2,
        ),
    )


def test_structured_response_uses_actual_citation_numbers():
    items = [
        StructuredItem(
            name="Computer Science and Design",
            chunk_id="chunk-003",
            source_url="https://example.com/design.pdf",
            source_title="Design Brochure",
        ),
    ]

    citations = [
        SourceCitation(
            number=1,
            title="Engineering Brochure",
            url="https://example.com/btech.pdf",
            chunk_id="chunk-001",
        ),
        SourceCitation(
            number=2,
            title="Design Brochure",
            url="https://example.com/design.pdf",
            chunk_id="chunk-003",
        ),
    ]

    structured = RAGPipeline._build_structured_answer(
        items,
        citations,
    )

    assert structured is not None
    assert structured.programmes[0].source_number == 2


def test_structured_response_ignores_items_without_citations():
    items = [
        StructuredItem(
            name="Computer Science and Engineering",
            chunk_id="missing-chunk",
            source_url=None,
            source_title="Engineering Brochure",
        ),
    ]

    structured = RAGPipeline._build_structured_answer(
        items,
        [],
    )

    assert structured is None


def test_structured_response_returns_none_for_empty_items():
    structured = RAGPipeline._build_structured_answer(
        [],
        [],
    )

    assert structured is None
