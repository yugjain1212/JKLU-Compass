from types import SimpleNamespace

from jklu_compass.rag.pipeline import RAGPipeline, SourceCitation


def make_item(name, chunk_id):
    return SimpleNamespace(
        name=name,
        chunk_id=chunk_id,
    )


def test_detects_missing_programmes():
    items = [
        make_item("Computer Science and Engineering", "chunk_1"),
        make_item(
            "Computer Science and Engineering (Artificial Intelligence)",
            "chunk_2",
        ),
    ]

    answer = "- Computer Science and Engineering (Artificial Intelligence) [Source 2]"

    missing = RAGPipeline._find_missing_programmes(answer, items)

    assert len(missing) == 1
    assert missing[0].name == "Computer Science and Engineering"


def test_detects_complete_programme_list():
    items = [
        make_item("Computer Science and Engineering", "chunk_1"),
        make_item("Computer Science and Design", "chunk_2"),
    ]

    answer = """
    - Computer Science and Engineering [Source 1]
    - Computer Science and Design [Source 2]
    """

    missing = RAGPipeline._find_missing_programmes(answer, items)

    assert missing == []


def test_appends_missing_programmes_with_correct_citations():
    items = [
        make_item("Computer Science and Design", "chunk_2"),
    ]

    citations = [
        SourceCitation(
            number=1,
            title="JKLU Brochure",
            url="https://example.com/brochure.pdf",
            chunk_id="chunk_1",
        ),
        SourceCitation(
            number=2,
            title="JKLU Brochure",
            url="https://example.com/brochure.pdf",
            chunk_id="chunk_2",
        ),
    ]

    answer = "The available programmes include other options."

    corrected = RAGPipeline._append_missing_programmes(
        answer,
        items,
        citations,
    )

    assert "Computer Science and Design" in corrected
    assert "[Source 2]" in corrected
