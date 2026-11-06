"""Tests for source-aware document retrieval."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from jklu_compass.retrieval import DocumentRetriever
from jklu_compass.retrieval.bm25_retriever import BM25Result


@pytest.fixture
def encoder() -> MagicMock:
    """Provide a fake encoder without loading a real model."""
    mock = MagicMock()
    mock.encode.return_value = np.array([0.1, 0.2, 0.3])
    return mock


@pytest.fixture
def vector_store() -> MagicMock:
    """Provide a fake vector store with source-aware results."""
    store = MagicMock()
    store.search.return_value = [
        {
            "id": "fee-chunk-1",
            "document": "JKLU publishes fee information for its programmes.",
            "metadata": {
                "source": "/data/fee-structure.md",
                "source_url": "https://jklu.edu.in/admissions/fee-structure",
                "title": "JKLU Fee Structure",
                "file_name": "fee-structure.md",
            },
            "distance": 0.12,
        }
    ]
    return store


@pytest.fixture
def retriever(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> DocumentRetriever:
    """Create a retriever using mocked dependencies."""
    return DocumentRetriever(encoder=encoder, vector_store=vector_store)


def test_retrieve_returns_source_url_and_title(
    retriever: DocumentRetriever,
) -> None:
    results = retriever.retrieve("What are JKLU fees?")

    assert len(results) == 1
    assert results[0].title == "JKLU Fee Structure"
    assert results[0].source_url == ("https://jklu.edu.in/admissions/fee-structure")
    assert results[0].content == ("JKLU publishes fee information for its programmes.")
    assert results[0].distance == 0.12


def test_retrieve_encodes_query_and_searches(
    retriever: DocumentRetriever,
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    retriever.retrieve("  What are JKLU fees?  ", top_k=3)

    encoder.encode.assert_called_once_with("What are JKLU fees?")
    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=9,
    )


def test_retrieve_deduplicates_chunks_from_same_source(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": "page-a-1",
            "document": "Best matching chunk from page A.",
            "metadata": {
                "title": "Page A",
                "source_url": "https://jklu.edu.in/academics",
            },
            "distance": 0.10,
        },
        {
            "id": "page-a-2",
            "document": "Another chunk from page A.",
            "metadata": {
                "title": "Page A",
                "source_url": "https://jklu.edu.in/academics/",
            },
            "distance": 0.12,
        },
        {
            "id": "page-b-1",
            "document": "Relevant chunk from page B.",
            "metadata": {
                "title": "Page B",
                "source_url": "https://jklu.edu.in/admissions",
            },
            "distance": 0.15,
        },
    ]

    results = retriever.retrieve("JKLU academics", top_k=2)

    assert len(results) == 2
    assert [result.id for result in results] == [
        "page-a-1",
        "page-b-1",
    ]


def test_retrieve_deduplicates_urls_with_tracking_parameters(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": "page-a-1",
            "document": "Original result.",
            "metadata": {
                "source_url": "https://jklu.edu.in/academics?utm_source=test",
            },
            "distance": 0.10,
        },
        {
            "id": "page-a-2",
            "document": "Duplicate result.",
            "metadata": {
                "source_url": "https://jklu.edu.in/academics#programmes",
            },
            "distance": 0.12,
        },
    ]

    results = retriever.retrieve("JKLU academics", top_k=2)

    assert len(results) == 1
    assert results[0].id == "page-a-1"


def test_retrieve_rejects_empty_query(
    retriever: DocumentRetriever,
) -> None:
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("   ")


def test_retrieve_rejects_invalid_top_k(
    retriever: DocumentRetriever,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("JKLU fees", top_k=0)


def test_local_source_is_not_returned_as_clickable_url(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value[0]["metadata"].pop("source_url")

    results = retriever.retrieve("JKLU fees")

    assert results[0].source == "/data/fee-structure.md"
    assert results[0].source_url is None


@pytest.mark.parametrize(
    "candidate",
    [
        "javascript:alert(1)",
        "file:///tmp/fee-structure.md",
        "not-a-url",
        "",
    ],
)
def test_invalid_source_url_is_rejected(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
    candidate: str,
) -> None:
    vector_store.search.return_value[0]["metadata"]["source_url"] = candidate

    results = retriever.retrieve("JKLU fees")

    assert results[0].source_url is None


def test_http_source_is_used_when_source_url_is_missing(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    metadata: dict[str, Any] = vector_store.search.return_value[0]["metadata"]
    metadata.pop("source_url")
    metadata["source"] = "https://jklu.edu.in/academics"

    results = retriever.retrieve("JKLU academics")

    assert results[0].source_url == "https://jklu.edu.in/academics"


def test_retrieve_prioritizes_matching_section_heading(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": "generic-1",
            "document": "JKLU offers several academic opportunities.",
            "metadata": {
                "source": "/data/about.md",
                "title": "About JKLU",
                "section_title": "About the University",
                "heading_path": "About JKLU > Overview",
            },
            "distance": 0.10,
        },
        {
            "id": "programme-1",
            "document": "JKLU offers B.Tech, B.Des and BBA programmes.",
            "metadata": {
                "source": "/data/programmes.md",
                "title": "Academic Programmes",
                "section_title": "Academic Programmes",
                "heading_path": "Academics > Academic Programmes",
            },
            "distance": 0.20,
        },
    ]

    results = retriever.retrieve(
        "What undergraduate programmes does JKLU offer?",
        top_k=1,
    )

    assert results[0].id == "programme-1"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What B.Tech specialisations are available at JKLU?", True),
        ("Which B.Tech programmes are listed in the brochure?", True),
        ("What semester-study opportunities are available to B.Tech students?", False),
        ("List B.Tech semester exchange opportunities.", False),
    ],
)
def test_btech_programme_list_intent_is_specific(query: str, expected: bool) -> None:
    """Only explicit programme-list questions use structured retrieval."""
    assert DocumentRetriever.is_btech_programme_list_query(query) is expected


def test_retrieve_applies_source_diversity_after_ranking(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": "programme-1",
            "document": "JKLU offers undergraduate programmes.",
            "metadata": {
                "source": "/data/programmes.md",
                "section_title": "Academic Programmes",
            },
            "distance": 0.10,
        },
        {
            "id": "programme-2",
            "document": "JKLU offers B.Tech and B.Des programmes.",
            "metadata": {
                "source": "/data/programmes.md",
                "section_title": "Academic Programmes",
            },
            "distance": 0.12,
        },
        {
            "id": "mobility-1",
            "document": "JKLU offers student mobility opportunities.",
            "metadata": {
                "source": "/data/mobility.md",
                "section_title": "Student Mobility",
            },
            "distance": 0.15,
        },
    ]

    results = retriever.retrieve(
        "What undergraduate programmes does JKLU offer?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].id == "programme-1"
    assert results[1].id == "mobility-1"


def test_retrieve_prioritizes_exact_degree_fee_evidence(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    """Keep the matching degree and year row when source diversity would drop it."""
    fee_url = "https://jklu.edu.in/admissions/fee-structure"
    source = "/data/fee-structure.md"
    generic_fee = {
        "id": "generic-fee-row",
        "document": (
            "### Tuition Fee for Academic Year 2026-27\n"
            "Academic Fee (Annual): 2,84,000 | Technology Fee (Annual): 10,000"
        ),
        "metadata": {
            "source": source,
            "source_url": fee_url,
            "section_title": "Tuition Fee for Academic Year 2026-27",
            "title": "JKLU Fee Structure",
        },
        "distance": 0.10,
    }
    exact_fee = {
        "id": "btech-fee-row",
        "document": (
            "### Tuition Fee for Academic Year 2026-27\n"
            "| Programme: **Bachelor of Technology** | Academic Fee (Annual): 2,84,000 | "
            "Technology Fee (Annual): 10,000 |"
        ),
        "metadata": {
            "source": source,
            "source_url": fee_url,
            "section_title": "Tuition Fee for Academic Year 2026-27",
            "title": "JKLU Fee Structure",
        },
        "distance": 0.12,
    }
    unrelated = {
        "id": "unrelated-programme",
        "document": "JKLU offers B.Tech programmes and student activities.",
        "metadata": {
            "source": "/data/about.md",
            "source_url": "https://jklu.edu.in/about",
            "section_title": "About JKLU",
            "title": "About JKLU",
        },
        "distance": 0.14,
    }
    vector_store.search.return_value = [generic_fee, unrelated, exact_fee]
    bm25_retriever = MagicMock()
    bm25_retriever.records = [generic_fee, unrelated, exact_fee]
    bm25_retriever.retrieve.return_value = [
        BM25Result(id="generic-fee-row", score=3.0),
        BM25Result(id="unrelated-programme", score=2.0),
        BM25Result(id="btech-fee-row", score=1.0),
    ]
    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )
    trace: dict[str, Any] = {}

    results = retriever.retrieve(
        "What is the annual B.Tech academic fee for 2026-27?",
        top_k=2,
        trace=trace,
    )

    assert results[0].id == "btech-fee-row"
    assert trace["selection_reasons"]["btech-fee-row"] == "selected_degree_fee_evidence"


@pytest.mark.parametrize(
    ("query", "specific_content", "expected_reason"),
    [
        (
            "Which specialised laboratories are listed by JKLU's Institute of Engineering and "
            "Technology?",
            "State-of-the-art facilities include the AI Lab, Data Science Lab, Computer Network "
            "and Cyber Security Lab, AR/VR Lab, Smart Grid Lab, Circuit Design and VLSI Lab, "
            "Embedded System and IoT Lab, Robotics and Automation Lab, and Software Engineering "
            "Lab.",
            "selected_comprehensive_lab_list",
        ),
        (
            "What types of hostel rooms does the JKLU Hostel Life Policy describe?",
            "The policy offers fully furnished double/Triple occupancy rooms with AC and non-AC "
            "options.",
            "selected_hostel_room_type_evidence",
        ),
    ],
)
def test_retrieve_prioritizes_specific_multi_answer_evidence(
    encoder: MagicMock,
    vector_store: MagicMock,
    query: str,
    specific_content: str,
    expected_reason: str,
) -> None:
    """Preserve a complete list or room-type passage when diversity would drop it."""
    source_url = (
        "https://jklu.edu.in/engineering" if "laborator" in query else "https://jklu.edu.in/hostel"
    )
    specific = {
        "id": "specific-evidence",
        "document": specific_content,
        "metadata": {"source_url": source_url, "section_title": "Policy evidence"},
        "distance": 0.12,
    }
    generic = {
        "id": "generic-same-source",
        "document": "JKLU facilities provide student support information.",
        "metadata": {"source_url": source_url, "section_title": "Overview"},
        "distance": 0.10,
    }
    unrelated = {
        "id": "unrelated-source",
        "document": "JKLU campus information and academic opportunities.",
        "metadata": {"source_url": "https://jklu.edu.in/about", "section_title": "About"},
        "distance": 0.11,
    }
    vector_store.search.return_value = [generic, unrelated, specific]
    bm25_retriever = MagicMock()
    bm25_retriever.records = [generic, unrelated, specific]
    bm25_retriever.retrieve.return_value = [
        BM25Result(id="generic-same-source", score=3.0),
        BM25Result(id="unrelated-source", score=2.0),
        BM25Result(id="specific-evidence", score=1.0),
    ]
    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )
    trace: dict[str, Any] = {}

    results = retriever.retrieve(query, top_k=2, trace=trace)

    assert results[0].id == "specific-evidence"
    assert trace["selection_reasons"]["specific-evidence"] == expected_reason


def test_specific_lab_question_does_not_prioritize_a_multi_lab_inventory() -> None:
    """A question about one lab's equipment needs that lab's detailed chunk."""
    retriever = DocumentRetriever(
        encoder=MagicMock(),
        vector_store=MagicMock(),
    )
    candidate = retriever._build_result(
        {
            "id": "lab-inventory",
            "document": (
                "The AI Lab, Data Science Lab, Cyber Security Lab, AR/VR Lab, and Smart Grid Lab."
            ),
            "metadata": {},
            "distance": 1.0,
        },
    )[0]

    assert (
        retriever._priority_evidence_reason(
            "What hardware and platforms are available in the Embedded System and IoT Lab?",
            candidate,
        )
        is None
    )


def test_retrieve_allows_multiple_chunks_from_pdf(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": f"pdf-chunk-{index}",
            "document": f"B.Tech programme information {index}.",
            "metadata": {
                "title": "Engineering Brochure",
                "source_url": "https://jklu.edu.in/uploads/engineering-brochure.pdf",
                "section_title": f"B.Tech Programme {index}",
            },
            "distance": 0.10 + index * 0.01,
        }
        for index in range(1, 7)
    ]

    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
        max_chunks_per_pdf=6,
    )

    results = retriever.retrieve("B.Tech programmes", top_k=6)

    assert len(results) == 6
    assert {result.id for result in results} == {f"pdf-chunk-{index}" for index in range(1, 7)}


def test_retrieve_keeps_html_source_limit_of_one(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": f"html-chunk-{index}",
            "document": f"JKLU academic information {index}.",
            "metadata": {
                "title": "Academic Programmes",
                "source_url": "https://jklu.edu.in/academics",
            },
            "distance": 0.10 + index * 0.01,
        }
        for index in range(1, 4)
    ]

    results = retriever.retrieve("JKLU academic information", top_k=3)

    assert len(results) == 1
    assert results[0].id == "html-chunk-1"


def test_retrieve_allows_query_distinct_second_html_chunk(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": "profile",
            "document": "Dr. Akshat Agrawal is an Associate Professor.",
            "metadata": {
                "title": "Faculty Profile",
                "section_title": "Dr. Akshat Agrawal",
                "source_url": "https://jklu.edu.in/faculty/akshat-agrawal",
            },
            "distance": 0.10,
        },
        {
            "id": "research",
            "document": "His primary research interests include artificial intelligence.",
            "metadata": {
                "title": "Faculty Profile",
                "section_title": "Research Interests",
                "source_url": "https://jklu.edu.in/faculty/akshat-agrawal",
            },
            "distance": 0.12,
        },
    ]
    retriever = DocumentRetriever(encoder=encoder, vector_store=vector_store)
    trace: dict[str, Any] = {}

    results = retriever.retrieve(
        "What are Dr. Akshat Agrawal's primary research interests?",
        top_k=2,
        trace=trace,
    )

    assert [result.id for result in results] == ["profile", "research"]
    assert trace["selection_reasons"]["research"] == "selected_for_fill"


def test_retrieve_respects_configured_pdf_chunk_limit(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = [
        {
            "id": f"pdf-chunk-{index}",
            "document": f"Engineering programme information {index}.",
            "metadata": {
                "source_url": "https://jklu.edu.in/engineering-brochure.pdf?download=1",
            },
            "distance": 0.10 + index * 0.01,
        }
        for index in range(1, 5)
    ]

    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
        max_chunks_per_pdf=3,
    )

    results = retriever.retrieve("Engineering programme information", top_k=4)

    assert len(results) == 3


def test_retrieve_trace_handles_missing_bm25_records(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value = []
    bm25_retriever = MagicMock()
    bm25_retriever.records = []
    bm25_retriever.retrieve.return_value = [BM25Result(id="missing", score=1.0)]
    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )
    trace: dict[str, Any] = {}

    assert retriever.retrieve("JKLU fees", trace=trace) == []
    assert trace["missing_record_ids"] == ["missing"]
    assert trace["ranked_candidates"] == []


def test_retrieve_trace_deduplicates_candidate_ids(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    record = {
        "id": "same-id",
        "document": "JKLU fee structure information.",
        "metadata": {"source": "/data/fees.md"},
        "distance": 0.1,
    }
    vector_store.search.return_value = [record, record]
    bm25_retriever = MagicMock()
    bm25_retriever.records = [record]
    bm25_retriever.retrieve.return_value = [
        BM25Result(id="same-id", score=2.0),
        BM25Result(id="same-id", score=1.0),
    ]
    retriever = DocumentRetriever(
        encoder=encoder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )
    trace: dict[str, Any] = {}

    results = retriever.retrieve("JKLU fees", top_k=1, trace=trace)

    assert [result.id for result in results] == ["same-id"]
    assert trace["dense_candidates"] == [{"rank": 1, "id": "same-id"}]
    assert trace["bm25_candidates"] == [{"rank": 1, "id": "same-id"}]
    assert trace["ranked_candidates"][0]["rrf_score"] == pytest.approx(2 / 61)
