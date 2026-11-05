import json

import pytest

from jklu_compass.evaluation.evaluator import RetrievalEvaluator


class MockRetrievalResult:
    def __init__(self, chunk_id: str) -> None:
        self.id = chunk_id


class MockRetriever:
    def __init__(self, retrieved_ids: list[str]) -> None:
        self.retrieved_ids = retrieved_ids
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[MockRetrievalResult]:
        self.queries.append((query, top_k))
        return [MockRetrievalResult(chunk_id) for chunk_id in self.retrieved_ids]


@pytest.fixture
def benchmark_file(tmp_path):
    data = {
        "questions": [
            {
                "id": "TEST-001",
                "question": "Test retrieval question?",
                "category": "academics",
                "difficulty": "easy",
                "relevant_chunk_ids": ["chunk-1", "chunk-2"],
                "expected_answer": ["Expected answer"],
            }
        ]
    }

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_evaluator_calculates_metrics(benchmark_file, tmp_path):
    retriever = MockRetriever(["chunk-1", "chunk-3"])

    evaluator = RetrievalEvaluator(
        retriever=retriever,
        benchmark_path=benchmark_file,
        results_dir=tmp_path / "results",
        top_k=2,
    )

    report = evaluator.evaluate(save=False)

    assert report["total_questions"] == 1
    assert report["aggregate_metrics"]["precision_at_k"] == 0.5
    assert report["aggregate_metrics"]["recall_at_k"] == 0.5
    assert report["aggregate_metrics"]["hit_rate_at_k"] == 1.0
    assert report["aggregate_metrics"]["mean_reciprocal_rank"] == 1.0


def test_evaluator_saves_json_report(benchmark_file, tmp_path):
    retriever = MockRetriever(["chunk-1", "chunk-2"])
    results_dir = tmp_path / "results"

    evaluator = RetrievalEvaluator(
        retriever=retriever,
        benchmark_path=benchmark_file,
        results_dir=results_dir,
        top_k=2,
    )

    report = evaluator.evaluate(save=True)

    report_path = report["report_path"]

    assert results_dir.exists()
    assert report_path.endswith(".json")

    saved_report = json.loads(open(report_path, encoding="utf-8").read())

    assert saved_report["total_questions"] == 1
    assert saved_report["aggregate_metrics"]["recall_at_k"] == 1.0


def test_evaluator_rejects_invalid_top_k(benchmark_file):
    with pytest.raises(ValueError, match="top_k"):
        RetrievalEvaluator(
            retriever=MockRetriever([]),
            benchmark_path=benchmark_file,
            top_k=0,
        )


def test_evaluator_does_not_save_when_disabled(benchmark_file, tmp_path):
    results_dir = tmp_path / "results"

    evaluator = RetrievalEvaluator(
        retriever=MockRetriever(["chunk-1"]),
        benchmark_path=benchmark_file,
        results_dir=results_dir,
    )

    evaluator.evaluate(save=False)

    assert not results_dir.exists()
