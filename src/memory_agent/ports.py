"""Public dependency-injection contracts for memory components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import MemoryRecord, RetrievalEvidence, SearchResult


@runtime_checkable
class MemoryStorePort(Protocol):
    """Persistence operations required by the memory lifecycle."""

    def add_memory(self, memory: MemoryRecord, *, commit: bool = True) -> int:
        """Persist a memory and return its identifier."""
        ...

    def get_memory(self, memory_id: int) -> MemoryRecord | None:
        """Return a memory by identifier, or ``None`` when absent."""
        ...

    def update_memory(self, memory: MemoryRecord, *, commit: bool = True) -> None:
        """Persist changes to an existing memory."""
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """Text embedding provider used to index and compare memories."""

    model_name: str

    def encode(self, text: str) -> bytes:
        """Encode text into a provider-specific serialized vector."""
        ...


@runtime_checkable
class RetrievalPort(Protocol):
    """Candidate retrieval operation exposed to the orchestrator."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        memory_type: str | None = None,
        min_score: float | None = None,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        """Return ranked candidates for a query."""
        ...


@runtime_checkable
class TrustPolicyPort(Protocol):
    """Trust decision operation applied before context injection."""

    def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
        """Explain whether a memory is trusted for retrieval."""
        ...
