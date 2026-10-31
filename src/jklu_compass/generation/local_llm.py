"""Local LLM client for JKLU Compass using Ollama."""

from __future__ import annotations

import requests


class LocalLLMError(RuntimeError):
    """Raised when the local LLM cannot generate a response."""


class LocalLLM:
    """Generate grounded responses using a locally running Ollama model."""

    def __init__(
        self,
        model: str = "qwen3.5:2b-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Generate a response from a prompt using the local Ollama API."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

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
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError as exc:
            raise LocalLLMError(
                "Cannot connect to Ollama. Make sure Ollama is running."
            ) from exc

        except requests.exceptions.Timeout as exc:
            raise LocalLLMError(
                f"Ollama timed out after {self.timeout} seconds."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            raise LocalLLMError(
                f"Ollama returned an HTTP error: {exc}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise LocalLLMError(
                f"Request to Ollama failed: {exc}"
            ) from exc

        generated_text = data.get("response", "")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise LocalLLMError(
                "Ollama returned an empty response."
            )

        return generated_text.strip() 