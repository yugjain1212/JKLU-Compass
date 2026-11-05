"""Read-only retrieval evaluation for the JKLU Compass benchmark."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jklu_compass.evaluation.dataset import load_benchmark
from jklu_compass.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


class RetrievalEvaluator:
    """Evaluate an existing document retriever against a benchmark dataset."""

    def __init__(
        self,
        retriever: Any,
        benchmark_path: str | Path,
        results_dir: str | Path = "data/evaluation/results",
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.retriever = retriever
        self.benchmark_path = Path(benchmark_path)
        self.results_dir = Path(results_dir)
        self.top_k = top_k

    def evaluate(self, save: bool = True) -> dict[str, Any]:
        """Run retrieval evaluation and optionally save a JSON report."""
        questions = load_benchmark(self.benchmark_path)

        question_results: list[dict[str, Any]] = []

        for question in questions:
            retrieved = self.retriever.retrieve(
                question.question,
                top_k=self.top_k,
            )

            retrieved_ids = [result.id for result in retrieved]
            relevant_ids = list(question.relevant_chunk_ids)

            metrics = {
                "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, self.top_k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, self.top_k),
                "hit_rate_at_k": hit_rate_at_k(retrieved_ids, relevant_ids, self.top_k),
                "mean_reciprocal_rank": mean_reciprocal_rank(retrieved_ids, relevant_ids),
            }

            question_results.append(
                {
                    "question_id": question.id,
                    "question": question.question,
                    "category": question.category,
                    "difficulty": question.difficulty,
                    "relevant_chunk_count": len(relevant_ids),
                    "retrieved_chunk_ids": retrieved_ids,
                    "metrics": metrics,
                }
            )

        metric_names = (
            "precision_at_k",
            "recall_at_k",
            "hit_rate_at_k",
            "mean_reciprocal_rank",
        )

        aggregate = {
            metric: round(
                sum(result["metrics"][metric] for result in question_results)
                / len(question_results),
                4,
            )
            for metric in metric_names
        }

        report: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "top_k": self.top_k,
            "total_questions": len(question_results),
            "aggregate_metrics": aggregate,
            "question_results": question_results,
        }

        if save:
            self.results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_path = self.results_dir / (f"retrieval_evaluation_{timestamp}.json")

            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            report["report_path"] = str(report_path)

        return report
