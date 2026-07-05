"""Tests for embedding fallback behavior."""

from __future__ import annotations

import pickle

import numpy as np

from memory_agent.core.embeddings import EmbeddingEngine


class UnavailableModel:
    def encode(self, *args, **kwargs):
        raise RuntimeError("model unavailable")


class TestEmbeddingEngineFallback:
    def test_encode_falls_back_to_deterministic_nonzero_float32_vector_when_model_unavailable(self):
        """A single encode failure should still produce a stable usable vector."""
        engine = EmbeddingEngine()
        engine._model = UnavailableModel()
        engine._dimension = 12

        first_blob = engine.encode("same text")
        second_blob = engine.encode("same text")
        different_blob = engine.encode("different text")

        first = pickle.loads(first_blob)
        second = pickle.loads(second_blob)
        different = pickle.loads(different_blob)

        assert isinstance(first_blob, bytes)
        assert first.dtype == np.float32
        assert first.shape == (engine.dimension,)
        assert np.linalg.norm(first) > 0
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, different)

    def test_encode_multiple_falls_back_per_text_when_batch_model_unavailable(self):
        """A batch encode failure should return stable usable vectors for every text."""
        engine = EmbeddingEngine()
        engine._model = UnavailableModel()
        engine._dimension = 10

        blobs = engine.encode_multiple(["alpha", "beta", "alpha"])

        vectors = [pickle.loads(blob) for blob in blobs]

        assert len(blobs) == 3
        assert all(isinstance(blob, bytes) for blob in blobs)
        assert all(vector.dtype == np.float32 for vector in vectors)
        assert all(vector.shape == (engine.dimension,) for vector in vectors)
        assert all(np.linalg.norm(vector) > 0 for vector in vectors)
        np.testing.assert_array_equal(vectors[0], vectors[2])
        assert not np.array_equal(vectors[0], vectors[1])
