"""Tests for the JKLU Compass API."""

from fastapi.testclient import TestClient

from jklu_compass.api.main import create_app
from jklu_compass.rag.pipeline import (
    RAGResponse,
    SourceCitation,
    StructuredAnswer,
    StructuredProgramme,
)


class FakePipeline:
    """A lightweight pipeline for API tests."""

    def ask(self, question: str, history: tuple[tuple[str, str], ...] = ()) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer=(
                "JKLU offers Computer Science and Engineering "
                "and Computer Science and Engineering (Artificial Intelligence)."
            ),
            sources=(
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
            ),
            retrieved_chunks=2,
            structured_data=StructuredAnswer(
                answer_type="btech_programme_list",
                programmes=(
                    StructuredProgramme(
                        name="Computer Science and Engineering",
                        source_number=1,
                    ),
                    StructuredProgramme(
                        name="Computer Science and Engineering (Artificial Intelligence)",
                        source_number=2,
                    ),
                ),
            ),
        )


def create_test_client() -> TestClient:
    """Create an API client with a fake RAG pipeline."""
    app = create_app(pipeline_factory=FakePipeline)
    return TestClient(app)


def test_health_endpoint() -> None:
    with create_test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline": "ready",
    }


def test_query_endpoint_returns_answer() -> None:
    with create_test_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "What B.Tech programmes are available?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What B.Tech programmes are available?"
    assert "Computer Science and Engineering" in data["answer"]
    assert data["retrieved_chunks"] == 2
    assert len(data["sources"]) == 2


def test_query_endpoint_returns_structured_data() -> None:
    with create_test_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "List B.Tech specialisations at JKLU."},
        )

    assert response.status_code == 200

    structured_data = response.json()["structured_data"]

    assert structured_data["answer_type"] == "btech_programme_list"
    assert len(structured_data["programmes"]) == 2

    assert structured_data["programmes"][0] == {
        "name": "Computer Science and Engineering",
        "source_number": 1,
    }

    assert structured_data["programmes"][1]["source_number"] == 2


def test_query_endpoint_rejects_empty_question() -> None:
    with create_test_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "   "},
        )

    assert response.status_code == 422


def test_query_endpoint_rejects_missing_question() -> None:
    with create_test_client() as client:
        response = client.post(
            "/api/v1/query",
            json={},
        )

    assert response.status_code == 422


def test_query_endpoint_supports_missing_structured_data() -> None:
    class BasicFakePipeline:
        def ask(self, question: str, history: tuple[tuple[str, str], ...] = ()) -> RAGResponse:
            return RAGResponse(
                question=question,
                answer="JKLU is a university in Jaipur.",
                sources=(),
                retrieved_chunks=0,
                structured_data=None,
            )

    app = create_app(pipeline_factory=BasicFakePipeline)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "Tell me about JKLU."},
        )

    assert response.status_code == 200
    assert response.json()["structured_data"] is None


class ClarifyingFakePipeline:
    """Fake pipeline that exposes the same shared LLM used by clarification."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        self.llm = FakeClarifierLLM()

    def ask(self, question: str, history: tuple[tuple[str, str], ...] = ()) -> RAGResponse:
        self.calls.append((question, history))
        return RAGResponse(
            question=question,
            answer="A grounded test answer. [Source 1]",
            sources=(SourceCitation(1, "Official page", "https://example.com", "chunk"),),
            retrieved_chunks=1,
        )


class FakeClarifierLLM:
    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        return (
            '{"clarification_required":true,"message":"Which area do you mean?",'
            '"options":["I mean the IET dean","I mean the Dean of Academic Affairs"]}'
        )


def test_query_clarification_selection_is_validated_and_single_use() -> None:
    with TestClient(create_app(pipeline_factory=ClarifyingFakePipeline)) as client:
        first = client.post(
            "/api/v1/query",
            json={
                "question": "Who is the dean?",
                "conversation_id": "conversation-1",
                "history": [{"role": "user", "content": "I study engineering."}],
            },
        )
        assert first.status_code == 200
        clarification = first.json()
        assert clarification["status"] == "clarification_required"
        assert len(clarification["options"]) == 2
        assert client.app.state.rag_pipeline.calls == []

        invalid = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "conversation-1",
                "clarification_id": clarification["clarification_id"],
                "option_id": "not-an-option",
            },
        )
        assert invalid.status_code == 422

        selected = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "conversation-1",
                "clarification_id": clarification["clarification_id"],
                "option_id": clarification["options"][0]["id"],
            },
        )
        assert selected.status_code == 200
        assert selected.json()["status"] == "answer"
        assert "Who is the dean?" in selected.json()["question"]
        assert "I mean the IET dean" in selected.json()["question"]
        assert client.app.state.rag_pipeline.calls[0][1] == (
            ("user", "I study engineering."),
        )
        duplicate = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "conversation-1",
                "clarification_id": clarification["clarification_id"],
                "option_id": clarification["options"][0]["id"],
            },
        )
        assert duplicate.status_code == 409


def test_new_clarification_makes_previous_selection_stale() -> None:
    with TestClient(create_app(pipeline_factory=ClarifyingFakePipeline)) as client:
        first = client.post(
            "/api/v1/query",
            json={"question": "Who is the dean?", "conversation_id": "same-chat"},
        ).json()
        second = client.post(
            "/api/v1/query",
            json={"question": "Which dean leads IET?", "conversation_id": "same-chat"},
        ).json()
        assert second["status"] == "clarification_required"
        stale = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "same-chat",
                "clarification_id": first["clarification_id"],
                "option_id": first["options"][0]["id"],
            },
        )
    assert stale.status_code == 409


def test_other_option_accepts_custom_clarification_text() -> None:
    with TestClient(create_app(pipeline_factory=ClarifyingFakePipeline)) as client:
        clarification = client.post(
            "/api/v1/query",
            json={"question": "Who is the dean?", "conversation_id": "custom-chat"},
        ).json()
        missing_text = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "custom-chat",
                "clarification_id": clarification["clarification_id"],
                "option_id": "other",
            },
        )
        assert missing_text.status_code == 422

        custom = "I mean the dean of JKLU's Institute of Engineering and Technology."
        response = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "custom-chat",
                "clarification_id": clarification["clarification_id"],
                "option_id": "other",
                "custom_answer": custom,
            },
        )
        assert response.status_code == 200
        assert custom in response.json()["question"]
        assert custom in client.app.state.rag_pipeline.calls[0][0]


def test_other_option_rejects_blank_or_oversized_custom_text() -> None:
    with TestClient(create_app(pipeline_factory=ClarifyingFakePipeline)) as client:
        clarification = client.post(
            "/api/v1/query",
            json={"question": "Who is the dean?", "conversation_id": "bounded-chat"},
        ).json()
        base = {
            "conversation_id": "bounded-chat",
            "clarification_id": clarification["clarification_id"],
            "option_id": "other",
        }
        blank = client.post("/api/v1/query", json={**base, "custom_answer": "  "})
        long = client.post("/api/v1/query", json={**base, "custom_answer": "x" * 501})
    assert blank.status_code == 422
    assert long.status_code == 422


def test_query_clarification_rejects_expired_selection() -> None:
    with TestClient(
        create_app(pipeline_factory=ClarifyingFakePipeline, clarification_ttl_seconds=0)
    ) as client:
        clarification = client.post(
            "/api/v1/query",
            json={"question": "Who is the dean?", "conversation_id": "expired"},
        ).json()
        response = client.post(
            "/api/v1/query",
            json={
                "conversation_id": "expired",
                "clarification_id": clarification["clarification_id"],
                "option_id": clarification["options"][0]["id"],
            },
        )
    assert response.status_code == 410
