"""Tests for the evaluation benchmark dataset."""

import json

import pytest

from jklu_compass.evaluation.dataset import load_benchmark


def test_load_benchmark(tmp_path):
    data = {
        "questions": [
            {
                "id": "TEST-001",
                "question": "What programmes are available?",
                "category": "academics",
                "difficulty": "medium",
                "relevant_chunk_ids": ["chunk-001", "chunk-002"],
                "expected_answer": ["Programme A"],
            }
        ]
    }

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    questions = load_benchmark(path)

    assert len(questions) == 1
    assert questions[0].id == "TEST-001"
    assert questions[0].relevant_chunk_ids == ("chunk-001", "chunk-002")
    assert questions[0].expected_answer == ("Programme A",)


def test_load_benchmark_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_benchmark(tmp_path / "missing.json")


def test_load_benchmark_rejects_empty_questions(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text('{"questions": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        load_benchmark(path)


def test_load_benchmark_rejects_duplicate_question_ids(tmp_path):
    question = {
        "id": "TEST-001",
        "question": "Test question?",
        "category": "academics",
        "difficulty": "easy",
        "relevant_chunk_ids": ["chunk-001"],
    }

    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps({"questions": [question, question]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_benchmark(path)


def test_load_benchmark_rejects_empty_relevant_ids(tmp_path):
    question = {
        "id": "TEST-001",
        "question": "Test question?",
        "category": "academics",
        "difficulty": "easy",
        "relevant_chunk_ids": [],
    }

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"questions": [question]}), encoding="utf-8")

    with pytest.raises(ValueError, match="relevant chunk IDs"):
        load_benchmark(path)
