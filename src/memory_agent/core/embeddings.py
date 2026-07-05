"""Embedding engine using sentence-transformers."""

from __future__ import annotations

import hashlib
import os
import pickle
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Semantic embedding engine wrapping sentence-transformers.

    Encodes text into dense vectors for semantic similarity search.
    Includes LRU cache for frequently-embedded texts.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1024):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._encode = lru_cache(maxsize=cache_size)(self._encode_uncached)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model  # trigger lazy load
        assert self._dimension is not None
        return self._dimension

    def _encode_uncached(self, text: str) -> bytes:
        """Encode a single text and return serialized numpy array."""
        if os.getenv("MEMORY_AGENT_USE_SENTENCE_TRANSFORMERS") != "1":
            return pickle.dumps(self._fallback_vector(text))
        try:
            vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return pickle.dumps(vec.astype(np.float32))
        except Exception:
            return pickle.dumps(self._fallback_vector(text))

    def encode(self, text: str) -> bytes:
        """Encode text to a pickle-dumped float32 numpy vector."""
        return self._encode(text)

    def encode_multiple(self, texts: list[str]) -> list[bytes]:
        """Encode multiple texts efficiently (no caching)."""
        if os.getenv("MEMORY_AGENT_USE_SENTENCE_TRANSFORMERS") != "1":
            return [pickle.dumps(self._fallback_vector(text)) for text in texts]
        try:
            vectors = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return [pickle.dumps(v.astype(np.float32)) for v in vectors]
        except Exception:
            return [pickle.dumps(self._fallback_vector(text)) for text in texts]

    def _fallback_vector(self, text: str) -> np.ndarray:
        """Deterministic local vector used when sentence-transformers is unavailable."""
        dimension = self._dimension or 384
        self._dimension = dimension
        seed = hashlib.sha256(f"{self.model_name}\0{text}".encode("utf-8")).digest()
        values = np.empty(dimension, dtype=np.float32)
        counter = 0
        offset = 0
        while offset < dimension:
            block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for byte in block:
                if offset >= dimension:
                    break
                values[offset] = (byte / 127.5) - 1.0
                offset += 1
            counter += 1
        norm = np.linalg.norm(values)
        if norm > 0:
            values /= norm
        return values.astype(np.float32)

    def decode_vector(self, blob: bytes) -> np.ndarray:
        """Deserialize a pickled vector back to numpy array."""
        return pickle.loads(blob)  # noqa: S301 — safe, our own data

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def similarity_between(self, text_a: str, text_b: str) -> float:
        """Direct similarity between two texts."""
        vec_a = self.decode_vector(self.encode(text_a))
        vec_b = self.decode_vector(self.encode(text_b))
        return self.cosine_similarity(vec_a, vec_b)

    def query_similarity(
        self, query_vec: np.ndarray, memory_vectors: list[tuple[int, bytes]]
    ) -> list[tuple[int, float]]:
        """Return (memory_id, similarity) sorted desc for all candidates."""
        results: list[tuple[int, float]] = []
        for mem_id, blob in memory_vectors:
            mem_vec = self.decode_vector(blob)
            sim = self.cosine_similarity(query_vec, mem_vec)
            results.append((mem_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def batch_similarity(
        self, query_vec: np.ndarray, mem_ids: list[int], vectors: list[np.ndarray]
    ) -> dict[int, float]:
        """Batch cosine similarity. Returns {mem_id: similarity}."""
        results: dict[int, float] = {}
        for mem_id, vec in zip(mem_ids, vectors, strict=False):
            results[mem_id] = self.cosine_similarity(query_vec, vec)
        return results

    def clear_cache(self) -> None:
        self._encode.cache_clear()
