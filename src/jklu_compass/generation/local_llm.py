"""Local LLM client for JKLU Compass using Ollama."""

from __future__ import annotations

import httpx


class LocalLLMError(RuntimeError):
    """Raised when the local LLM cannot generate a response."""


class LocalLLM:
    """Generate responses using a locally running Ollama model."""

    def __init__(
        self,
        model: str = "qwen3.5:2b-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        if not model.strip():
            raise ValueError("Model name cannot be empty.")

        if timeout < 1:
            raise ValueError("Timeout must be at least 1 second.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Generate a response using the Ollama generate API."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0 and 2.")

        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = httpx.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.ConnectError as exc:
            raise LocalLLMError("Cannot connect to Ollama. Make sure Ollama is running.") from exc

        except httpx.TimeoutException as exc:
            raise LocalLLMError(f"Ollama timed out after {self.timeout} seconds.") from exc

        except httpx.HTTPStatusError as exc:
            raise LocalLLMError(f"Ollama returned an HTTP error: {exc}") from exc

        except httpx.RequestError as exc:
            raise LocalLLMError(f"Request to Ollama failed: {exc}") from exc

        except ValueError as exc:
            raise LocalLLMError("Ollama returned an invalid JSON response.") from exc

        generated_text = data.get("response", "")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise LocalLLMError("Ollama returned an empty response.")

        return generated_text.strip()
