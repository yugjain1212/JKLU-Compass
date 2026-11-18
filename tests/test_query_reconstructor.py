"""Tests for reconstructing clarified search queries."""

from jklu_compass.rag.query_reconstructor import QueryReconstructor


def test_reconstruction_preserves_question_selection_and_history() -> None:
    query = QueryReconstructor.reconstruct(
        "What about his qualifications?",
        "I mean the current IET dean",
        (("user", "Who is the IET dean?"), ("assistant", "Dr. Tapas Kumar.")),
    )
    assert "What about his qualifications?" in query
    assert "I mean the current IET dean" in query
    assert "Dr. Tapas Kumar" in query


def test_reconstruction_works_without_conversation_history() -> None:
    query = QueryReconstructor.reconstruct("Who is the dean?", "The IET dean")
    assert "Who is the dean?" in query
    assert "The IET dean" in query
    assert "Relevant conversation" not in query
