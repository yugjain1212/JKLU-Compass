"""Load and validate the JKLU Compass evaluation benchmark."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    """A single question with manually verified relevant chunks."""

    id: str
    question: str
    category: str
    difficulty: str
    relevant_chunk_ids: tuple[str, ...]
    expected_answer: tuple[str, ...] = ()


def load_benchmark(path: str | Path) -> list[BenchmarkQuestion]:
    """Load benchmark questions from a JSON file."""
    benchmark_path = Path(path)

    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with benchmark_path.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    questions = data.get("questions")

    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark must contain a non-empty questions list.")

    results: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Every benchmark question must be an object.")

        question_id = item.get("id")
        question = item.get("question")
        category = item.get("category")
        difficulty = item.get("difficulty")
        relevant_ids = item.get("relevant_chunk_ids")
        expected_answer = item.get("expected_answer", [])

        if not all(
            isinstance(value, str) and value.strip()
            for value in (question_id, question, category, difficulty)
        ):
            raise ValueError("Question ID, question, category, and difficulty are required.")

        if question_id in seen_ids:
            raise ValueError(f"Duplicate benchmark question ID: {question_id}")

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(value, str) or not value.strip() for value in relevant_ids)
        ):
            raise ValueError(f"Question {question_id} must have relevant chunk IDs.")

        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(f"Question {question_id} contains duplicate relevant chunk IDs.")

        if not isinstance(expected_answer, list) or any(
            not isinstance(value, str) or not value.strip() for value in expected_answer
        ):
            raise ValueError(f"Question {question_id} has an invalid expected_answer.")

        seen_ids.add(question_id)

        results.append(
            BenchmarkQuestion(
                id=question_id,
                question=question,
                category=category,
                difficulty=difficulty,
                relevant_chunk_ids=tuple(relevant_ids),
                expected_answer=tuple(expected_answer),
            )
        )

    return results
