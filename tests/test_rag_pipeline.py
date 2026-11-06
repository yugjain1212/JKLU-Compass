"""Tests for the JKLU Compass RAG pipeline."""

from unittest.mock import MagicMock

import pytest

from jklu_compass.rag.pipeline import RAGPipeline
from jklu_compass.rag.structured_extractor import (
    StructuredExtractor,
    StructuredItem,
)
from jklu_compass.retrieval.retriever import RetrievalResult


@pytest.fixture
def retriever() -> MagicMock:
    """Provide a mocked document retriever."""
    return MagicMock()


@pytest.fixture
def llm() -> MagicMock:
    """Provide a mocked local language model."""
    mock = MagicMock()
    mock.generate.return_value = "JKLU offers several undergraduate programmes. [Source 1]"
    return mock


@pytest.fixture
def pipeline(
    retriever: MagicMock,
    llm: MagicMock,
) -> RAGPipeline:
    """Create a RAG pipeline with mocked dependencies."""
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=5,
    )


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Provide representative retrieved document chunks."""
    return [
        RetrievalResult(
            id="chunk-001",
            content="JKLU offers B.Tech programmes in engineering.",
            title="Engineering Brochure",
            source_url="https://jklu.edu.in/engineering.pdf",
            source="/data/engineering.md",
            distance=0.12,
            metadata={"section_title": "B.Tech Programmes"},
        ),
        RetrievalResult(
            id="chunk-002",
            content="JKLU offers undergraduate design programmes.",
            title="Design Brochure",
            source_url="https://jklu.edu.in/design.pdf",
            source="/data/design.md",
            distance=0.18,
            metadata={"section_title": "Design Programmes"},
        ),
    ]


def test_ask_retrieves_documents_and_generates_answer(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Retrieve context and generate a grounded answer."""
    retriever.retrieve.return_value = sample_results

    response = pipeline.ask("What programmes does JKLU offer?")

    retriever.retrieve.assert_called_once_with(
        "What programmes does JKLU offer?",
        top_k=5,
    )
    llm.generate.assert_called_once()

    assert response.question == "What programmes does JKLU offer?"
    assert response.answer == ("JKLU offers several undergraduate programmes. [Source 1]")
    assert response.retrieved_chunks == 2
    assert len(response.sources) == 2


def test_ask_returns_correct_source_citations(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Preserve citation numbers, URLs, titles, and chunk IDs."""
    retriever.retrieve.return_value = sample_results

    response = pipeline.ask("What programmes does JKLU offer?")

    assert response.sources[0].number == 1
    assert response.sources[0].title == "Engineering Brochure"
    assert response.sources[0].url == "https://jklu.edu.in/engineering.pdf"
    assert response.sources[0].chunk_id == "chunk-001"

    assert response.sources[1].number == 2
    assert response.sources[1].title == "Design Brochure"
    assert response.sources[1].chunk_id == "chunk-002"


def test_ask_does_not_call_llm_when_retrieval_is_empty(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """Return a fallback response when no documents are retrieved."""
    retriever.retrieve.return_value = []

    response = pipeline.ask("What are the JKLU fees?")

    llm.generate.assert_not_called()

    assert response.sources == ()
    assert response.retrieved_chunks == 0
    assert "couldn't find usable information" in response.answer


def test_ask_filters_empty_chunks_from_context(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Exclude empty chunks and maintain consecutive citation numbers."""
    empty_result = RetrievalResult(
        id="empty-chunk",
        content="   ",
        title="Empty document",
        source_url=None,
        source="/data/empty.md",
        distance=0.1,
        metadata={},
    )

    retriever.retrieve.return_value = [
        empty_result,
        sample_results[0],
    ]

    response = pipeline.ask("What does JKLU offer?")

    assert len(response.sources) == 1
    assert response.sources[0].number == 1
    assert response.sources[0].chunk_id == "chunk-001"

    prompt = llm.generate.call_args.args[0]
    assert "Empty document" not in prompt
    assert "[Source 1]" in prompt


def test_ask_includes_grounding_instructions(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Ensure the prompt contains evidence and citation rules."""
    retriever.retrieve.return_value = sample_results

    pipeline.ask("What programmes does JKLU offer?")

    prompt = llm.generate.call_args.args[0]

    assert "Never invent university facts" in prompt
    assert "reference passages as evidence" in prompt
    assert "Do not invent citations or URLs" in prompt
    assert "include only information for that degree" in prompt
    assert "Only claim a requested academic year" in prompt
    assert "Engineering Brochure" in prompt
    assert "Design Brochure" in prompt


def test_ask_uses_low_temperature(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Use a low temperature for factual university responses."""
    retriever.retrieve.return_value = sample_results

    pipeline.ask("What programmes does JKLU offer?")

    llm.generate.assert_called_once()
    assert llm.generate.call_args.kwargs["temperature"] == 0.1


def test_ask_rejects_empty_question(
    pipeline: RAGPipeline,
) -> None:
    """Reject invalid questions before retrieval."""
    with pytest.raises(ValueError, match="(?i)question"):
        pipeline.ask("   ")


def test_ask_normalizes_question_whitespace(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Strip surrounding whitespace before retrieval."""
    retriever.retrieve.return_value = sample_results

    response = pipeline.ask("  What programmes does JKLU offer?  ")

    assert response.question == "What programmes does JKLU offer?"
    retriever.retrieve.assert_called_once_with(
        "What programmes does JKLU offer?",
        top_k=5,
    )


def test_ask_ignores_invalid_source_titles(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    sample_results: list[RetrievalResult],
) -> None:
    """Replace invalid document titles with a safe fallback."""
    invalid_title_result = RetrievalResult(
        id="chunk-003",
        content="Some relevant JKLU information.",
        title="undefined",
        source_url=None,
        source="/data/unknown.md",
        distance=0.2,
        metadata={},
    )

    retriever.retrieve.return_value = [
        invalid_title_result,
        sample_results[0],
    ]

    response = pipeline.ask("Tell me about JKLU.")

    assert response.sources[0].title == "Untitled JKLU document"


def test_btech_programme_question_uses_deterministic_answer(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render all six verified B.Tech programmes without calling the LLM."""

    programme_names = [
        "Computer Science and Engineering",
        "Computer Science and Engineering (Artificial Intelligence)",
        "Computer Science and Engineering (Data Science)",
        "Computer Science and Engineering (Cyber Security)",
        "Computer Science and Design",
        "Electronics and Communication Engineering (VLSI Design)",
    ]

    items = [
        StructuredItem(
            name=name,
            chunk_id=f"chunk-{index:03d}",
            source_url="https://jklu.edu.in/engineering.pdf",
            source_title="Engineering Brochure",
        )
        for index, name in enumerate(programme_names, start=1)
    ]

    retrieved_results = [
        RetrievalResult(
            id=item.chunk_id,
            content=f"JKLU B.Tech programme: {item.name}",
            title=item.source_title,
            source_url=item.source_url,
            source="/data/engineering.md",
            distance=0.1 + index * 0.01,
            metadata={"section_title": item.name},
        )
        for index, item in enumerate(items)
    ]

    retriever.retrieve.return_value = retrieved_results

    monkeypatch.setattr(
        StructuredExtractor,
        "extract_btech_programmes",
        staticmethod(lambda results: items),
    )

    response = pipeline.ask("What B.Tech specialisations are available at JKLU?")

    # The deterministic path must bypass LLM generation.
    llm.generate.assert_not_called()

    # Verify structured response.
    assert response.structured_data is not None
    assert response.structured_data.answer_type == "btech_programme_list"

    structured_programmes = response.structured_data.programmes

    assert len(structured_programmes) == 6
    assert [programme.name for programme in structured_programmes] == programme_names
    assert [programme.source_number for programme in structured_programmes] == [1, 2, 3, 4, 5, 6]

    # Verify rendered answer.
    for index, name in enumerate(programme_names, start=1):
        expected_line = f"{index}. {name} [Source {index}]"
        assert response.answer.count(expected_line) == 1

    # Verify citations.
    assert len(response.sources) == 6

    for index, source in enumerate(response.sources, start=1):
        assert source.number == index
        assert source.chunk_id == f"chunk-{index:03d}"

    assert "current admissions catalogue" in response.answer
    assert response.retrieved_chunks == 6


def test_btech_semester_opportunity_question_uses_normal_generation(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    sample_results: list[RetrievalResult],
) -> None:
    """Do not mistake semester mobility for a B.Tech programme-list query."""
    retriever.retrieve.return_value = sample_results
    extract = MagicMock(return_value=[])
    monkeypatch.setattr(StructuredExtractor, "extract_btech_programmes", staticmethod(extract))

    response = pipeline.ask("What semester-study opportunities are available to B.Tech students?")

    extract.assert_not_called()
    llm.generate.assert_called_once()
    assert response.structured_data is None
    assert (
        response.question == "What semester-study opportunities are available to B.Tech students?"
    )
    assert [source.chunk_id for source in response.sources] == ["chunk-001"]


def test_single_degree_question_omits_mixed_degree_context(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """Keep the prompt focused when a single-degree chunk is available."""
    retriever.retrieve.return_value = [
        RetrievalResult(
            id="btech-fee",
            content="B.Tech annual academic fee is INR 284000.",
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.1,
            metadata={"section_title": "Bachelor of Technology"},
        ),
        RetrievalResult(
            id="combined-fees",
            content=(
                "# Fee Structure\n"
                "### Tuition Fee for Academic Year 2026-27\n"
                "| Programme | Academic Fee (Annual) | Technology Fee (Annual) |\n"
                "| --- | --- | --- |\n"
                "| Programme: **Bachelor of Design** | 3,51,000 | 30,000 |\n"
                "| Programme: **Bachelor of Technology** | 2,84,000 | 10,000 |"
            ),
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.12,
            metadata={},
        ),
        RetrievalResult(
            id="bdes-fee",
            content="B.Des annual academic fee is INR 351000.",
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.14,
            metadata={"section_title": "Bachelor of Design"},
        ),
        RetrievalResult(
            id="btech-scholarship",
            content="B.Tech scholarship eligibility for Class XII students.",
            title="Scholarship Aid",
            source_url="https://jklu.edu.in/scholarship",
            source="/data/scholarship.md",
            distance=0.16,
            metadata={"section_title": "B.Tech Scholarship"},
        ),
    ]

    response = pipeline.ask("What is the annual B.Tech academic fee for 2026-27?")

    llm.generate.assert_not_called()
    assert response.answer == (
        "For academic year 2026–27, B.Tech annual academic fee: ₹2,84,000 [Source 1]"
    )
    assert [source.chunk_id for source in response.sources] == ["combined-fees"]
    assert response.retrieved_chunks == 4


def test_generic_annual_degree_fee_returns_available_fee_fields(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """A general annual-fee question can return both explicit fee fields."""
    retriever.retrieve.return_value = [
        RetrievalResult(
            id="btech-fees",
            content=(
                "### Academic Year 2026-27\n"
                "Programme: Bachelor of Technology\n"
                "Academic Fee (Annual): 2,84,000\n"
                "Technology Fee (Annual): 10,000"
            ),
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.1,
            metadata={},
        ),
    ]

    response = pipeline.ask("What are the annual B.Tech fees for 2026-27?")

    llm.generate.assert_not_called()
    assert response.answer == (
        "For academic year 2026–27, B.Tech annual fees:\n"
        "- Annual academic fee: ₹2,84,000 [Source 1]\n"
        "- Annual technology fee: ₹10,000 [Source 1]"
    )
    assert [source.chunk_id for source in response.sources] == ["btech-fees"]


def test_yearless_fee_answer_discloses_the_source_year(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """Do not present a dated fee as current when the user omitted a year."""
    retriever.retrieve.return_value = [
        RetrievalResult(
            id="dated-btech-fee",
            content=(
                "### Academic Year 2026-27\n"
                "Programme: Bachelor of Technology\n"
                "Academic Fee (Annual): 2,84,000"
            ),
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.1,
            metadata={},
        ),
    ]

    response = pipeline.ask("What is the B.Tech annual academic fee?")

    llm.generate.assert_not_called()
    assert response.answer == (
        "For academic year 2026–27, B.Tech annual academic fee: ₹2,84,000 [Source 1]"
    )


def test_multi_year_fee_question_does_not_render_one_year_as_the_answer(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """Keep year comparisons in the generation path instead of dropping a year."""
    retriever.retrieve.return_value = [
        RetrievalResult(
            id="multi-year-fees",
            content=(
                "Programme: Bachelor of Technology\n"
                "2025-26 Academic Fee (Annual): 2,50,000\n"
                "2026-27 Academic Fee (Annual): 2,84,000"
            ),
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.1,
            metadata={},
        ),
    ]

    pipeline.ask("What were B.Tech academic fees in 2025-26 and 2026-27?")

    llm.generate.assert_called_once()
    prompt = llm.generate.call_args.args[0]
    assert "2025-26 Academic Fee" in prompt
    assert "2026-27 Academic Fee" in prompt


def test_degree_comparison_keeps_both_degrees_in_context(
    pipeline: RAGPipeline,
    retriever: MagicMock,
    llm: MagicMock,
) -> None:
    """Preserve mixed degree evidence when the user asks for a comparison."""
    retriever.retrieve.return_value = [
        RetrievalResult(
            id="combined-fees",
            content="B.Tech fee is INR 284000. B.Des fee is INR 351000.",
            title="Fee Structure",
            source_url="https://jklu.edu.in/fees",
            source="/data/fees.md",
            distance=0.12,
            metadata={},
        )
    ]

    response = pipeline.ask("Compare the annual B.Tech and B.Des academic fees.")

    prompt = llm.generate.call_args.args[0]
    assert "B.Tech fee is INR 284000" in prompt
    assert "B.Des fee is INR 351000" in prompt
    assert response.sources[0].chunk_id == "combined-fees"


def test_academic_year_labels_are_normalized() -> None:
    """Accept short and full academic-year suffixes and dash variants."""
    assert RAGPipeline._academic_years("2025–2026, 2026/27") == {"2025-26", "2026-27"}
