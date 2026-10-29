"""Text embedding utilities for JKLU Compass."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 384


class EmbeddingEncoder:
    """Encode text into dense semantic vectors using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the encoder configuration and load its model."""
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name)


        self.embedding_dimension = self._model.get_sentence_embedding_dimension()

        if self.embedding_dimension is None:
            raise RuntimeError("The embedding model did not report its output dimension.")

    def encode(self, text: str) -> NDArray:
        """Encode a single non-empty text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")

        embeddings = self.encode_batch([text])
        return embeddings[0]

    def encode_batch(self, texts: Sequence[str]) -> NDArray:
        """Encode multiple non-empty text strings into a two-dimensional array."""
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string.")

        if not texts:
            raise ValueError("texts must contain at least one string.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every item in texts must be a non-empty string.")

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embeddings
