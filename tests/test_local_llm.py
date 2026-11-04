"""Tests for the local Ollama LLM client."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from jklu_compass.generation.local_llm import LocalLLM, LocalLLMError


@pytest.fixture
def llm() -> LocalLLM:
    """Create a local LLM client for testing."""
    return LocalLLM(
        model="qwen3.5:2b-q4_K_M",
        base_url="http://localhost:11434",
        timeout=30,
    )


@patch("jklu_compass.generation.local_llm.httpx.post")
def test_generate_returns_response(
    mock_post: MagicMock,
    llm: LocalLLM,
) -> None:
    """Return generated text from a successful Ollama response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "RAG combines retrieval and generation."}
    mock_post.return_value = mock_response

    result = llm.generate("Explain RAG.")

    assert result == "RAG combines retrieval and generation."

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3.5:2b-q4_K_M",
            "prompt": "Explain RAG.",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2},
        },
        timeout=30,
    )

    mock_response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_generate_rejects_empty_prompt(
    llm: LocalLLM,
    prompt: str | None,
) -> None:
    """Reject empty or invalid prompts."""
    with pytest.raises(ValueError, match="Prompt"):
        llm.generate(prompt)  # type: ignore[arg-type]


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_generate_rejects_invalid_temperature(
    llm: LocalLLM,
    temperature: float,
) -> None:
    """Reject temperatures outside the supported range."""
    with pytest.raises(ValueError, match="Temperature"):
        llm.generate("Explain RAG.", temperature=temperature)


def test_init_rejects_empty_model() -> None:
    """Reject an empty model name."""
    with pytest.raises(ValueError, match="Model"):
        LocalLLM(model="   ")


def test_init_rejects_invalid_timeout() -> None:
    """Reject zero or negative timeouts."""
    with pytest.raises(ValueError, match="Timeout"):
        LocalLLM(timeout=0)


@patch("jklu_compass.generation.local_llm.httpx.post")
def test_generate_handles_connection_error(
    mock_post: MagicMock,
    llm: LocalLLM,
) -> None:
    """Raise a useful error when Ollama is unavailable."""
    mock_post.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(LocalLLMError, match="Cannot connect to Ollama"):
        llm.generate("Explain RAG.")


@patch("jklu_compass.generation.local_llm.httpx.post")
def test_generate_handles_timeout(
    mock_post: MagicMock,
    llm: LocalLLM,
) -> None:
    """Raise a useful error when Ollama times out."""
    mock_post.side_effect = httpx.ReadTimeout("Request timed out")

    with pytest.raises(LocalLLMError, match="timed out"):
        llm.generate("Explain RAG.")


@patch("jklu_compass.generation.local_llm.httpx.post")
def test_generate_handles_http_error(
    mock_post: MagicMock,
    llm: LocalLLM,
) -> None:
    """Handle HTTP status errors returned by Ollama."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error",
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        response=httpx.Response(500),
    )
    mock_post.return_value = mock_response

    with pytest.raises(LocalLLMError, match="HTTP error"):
        llm.generate("Explain RAG.")


@patch("jklu_compass.generation.local_llm.httpx.post")
def test_generate_handles_invalid_json(
    mock_post: MagicMock,
    llm: LocalLLM,
) -> None:
    """Handle invalid JSON returned by Ollama."""
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    with pytest.raises(LocalLLMError, match="invalid JSON"):
        llm.generate("Explain RAG.")


@pytest.mark.parametrize(
    "response_data",
    [
        {},
        {"response": ""},
        {"response": "   "},
        {"response": None},
        {"response": 123},
    ],
)
@patch("jklu_compass.generation.local_llm.httpx.post")
def test_generate_rejects_empty_response(
    mock_post: MagicMock,
    llm: LocalLLM,
    response_data: dict[str, object],
) -> None:
    """Reject empty or invalid generated responses."""
    mock_response = MagicMock()
    mock_response.json.return_value = response_data
    mock_post.return_value = mock_response

    with pytest.raises(LocalLLMError, match="empty response"):
        llm.generate("Explain RAG.")
