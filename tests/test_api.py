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

    def ask(self, question: str) -> RAGResponse:
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
        def ask(self, question: str) -> RAGResponse:
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
