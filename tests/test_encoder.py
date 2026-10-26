"""Tests for the embedding encoder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jklu_compass.embeddings.encoder import EmbeddingEncoder


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock embedding model."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.side_effect = lambda texts, **kwargs: np.ones(
        (len(texts), 384),
        dtype=np.float32,
    )
    return model


@pytest.fixture
def encoder(mock_model: MagicMock) -> EmbeddingEncoder:
    """Create an encoder without downloading a real model."""
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_model,
    ):
        instance = EmbeddingEncoder()

    return instance


def test_encoder_initializes_with_default_configuration() -> None:
    """The encoder should use the configured default model."""
    with patch("sentence_transformers.SentenceTransformer") as model_class:
        model_class.return_value.get_sentence_embedding_dimension.return_value = 384

        encoder = EmbeddingEncoder()

    model_class.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    assert encoder.embedding_dimension == 384
    assert encoder.batch_size == 32
    assert encoder.normalize_embeddings is True


def test_encode_returns_single_embedding(encoder: EmbeddingEncoder) -> None:
    """Encoding one string should return a one-dimensional vector."""
    result = encoder.encode("What programs does JKLU offer?")

    assert result.shape == (384,)


def test_encode_batch_returns_embedding_matrix(encoder: EmbeddingEncoder) -> None:
    """Encoding multiple strings should return a matrix."""
    result = encoder.encode_batch(
        [
            "JKLU admissions information",
            "JKLU academic programs",
        ]
    )

    assert result.shape == (2, 384)


def test_encode_rejects_empty_text(encoder: EmbeddingEncoder) -> None:
    """Single-text encoding should reject empty input."""
    with pytest.raises(ValueError, match="non-empty string"):
        encoder.encode("   ")


def test_encode_batch_rejects_empty_sequence(encoder: EmbeddingEncoder) -> None:
    """Batch encoding should reject an empty sequence."""
    with pytest.raises(ValueError, match="at least one string"):
        encoder.encode_batch([])


def test_encode_batch_rejects_single_string(encoder: EmbeddingEncoder) -> None:
    """Batch encoding should reject a string passed instead of a sequence."""
    with pytest.raises(TypeError, match="sequence of strings"):
        encoder.encode_batch("JKLU admissions")  # type: ignore[arg-type]


def test_encode_batch_rejects_empty_items(encoder: EmbeddingEncoder) -> None:
    """Every batch item must contain non-whitespace text."""
    with pytest.raises(ValueError, match="Every item"):
        encoder.encode_batch(["JKLU admissions", " "])


def test_encoder_rejects_invalid_batch_size() -> None:
    """Batch size must be a positive integer."""
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingEncoder(batch_size=0)


def test_encoder_rejects_empty_model_name() -> None:
    """Model name must not be empty."""
    with pytest.raises(ValueError, match="model_name"):
        EmbeddingEncoder(model_name="   ")
