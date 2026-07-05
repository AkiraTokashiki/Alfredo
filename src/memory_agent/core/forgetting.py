"""Ebbinghaus forgetting curve with importance-modulated decay."""

from __future__ import annotations

import math
from datetime import datetime

from memory_agent.core.config import ForgettingConfig
from memory_agent.models import MemoryRecord


class ForgettingCurve:
    """Ebbinghaus forgetting curve implementation.

    strength(t) = initial_strength * e^(-t / decay_constant)

    The decay constant varies by importance:
    - High importance (>= 0.8): decays over ~30 days
    - Medium importance (>= 0.5): decays over ~7 days
    - Low importance (< 0.5): decays over ~1 day

    Each retrieval reinforces the memory, boosting strength.
    When strength falls below archival_threshold, the memory is
    candidate for archival.
    """

    def __init__(self, config: ForgettingConfig | None = None):
        self.config = config or ForgettingConfig()

    # ------------------------------------------------------------------
    # Decay calculation
    # ------------------------------------------------------------------

    def decay_hours_for(self, importance: float) -> float:
        """Get the decay time constant in hours based on importance."""
        if importance >= self.config.importance_high:
            return self.config.decay_hours_high
        elif importance >= self.config.importance_medium:
            return self.config.decay_hours_medium
        else:
            return self.config.decay_hours_low

    def strength_after_decay(
        self, current_strength: float, importance: float, elapsed_hours: float
    ) -> float:
        """Compute strength after elapsed hours with Ebbinghaus decay.

        Args:
            current_strength: Current recall strength [0, 1].
            importance: Importance score [0, 1] — modulates decay rate.
            elapsed_hours: Hours since last update.

        Returns:
            New strength after decay.
        """
        decay_hours = self.decay_hours_for(importance)
        return current_strength * math.exp(-elapsed_hours / decay_hours)

    def apply_decay_to_memory(
        self, memory: MemoryRecord, now: datetime | None = None
    ) -> float:
        """Apply forgetting decay to a single memory record.

        Updates the memory's strength in-place.

        Returns:
            New strength value.
        """
        now = now or datetime.now()
        created = datetime.fromisoformat(memory.created_at) if memory.created_at else now

        # Use last_accessed_at if available, otherwise created_at
        reference = (
            datetime.fromisoformat(memory.last_accessed_at)
            if memory.last_accessed_at
            else created
        )

        elapsed_hours = max(0.0, (now - reference).total_seconds() / 3600)
        new_strength = self.strength_after_decay(
            memory.strength, memory.importance, elapsed_hours
        )
        memory.strength = new_strength
        return new_strength

    # ------------------------------------------------------------------
    # Reinforcement (retrieval boost)
    # ------------------------------------------------------------------

    def reinforce(self, memory: MemoryRecord) -> float:
        """Boost memory strength on retrieval.

        strength = min(1.0, strength + reinforcement_boost)

        Returns:
            New strength value.
        """
        memory.strength = min(
            self.config.max_strength,
            memory.strength + self.config.reinforcement_boost,
        )
        memory.access_count += 1
        memory.last_accessed_at = datetime.now().isoformat()
        return memory.strength

    # ------------------------------------------------------------------
    # Archival decision
    # ------------------------------------------------------------------

    def should_archive(self, memory: MemoryRecord) -> bool:
        """Check if memory strength has fallen below archival threshold."""
        return memory.strength < self.config.archival_threshold

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def decay_all(
        self, memories: list[MemoryRecord], now: datetime | None = None
    ) -> list[tuple[float, int]]:
        """Apply decay to all memories. Returns list of (new_strength, memory_id)."""
        now = now or datetime.now()
        updates: list[tuple[float, int]] = []
        for mem in memories:
            new_strength = self.apply_decay_to_memory(mem, now)
            assert mem.id is not None
            updates.append((new_strength, mem.id))
        return updates

    def reinforce_and_access(self, memory: MemoryRecord) -> None:
        """Reinforce strength and update access timestamp."""
        self.reinforce(memory)

    def predicted_lifespan_hours(self, importance: float) -> float:
        """Estimate how many hours until strength reaches archival threshold.

        Based on the equation: threshold = 1.0 * e^(-t / decay_hours)
        So t = -decay_hours * ln(threshold)
        """
        decay_hours = self.decay_hours_for(importance)
        return -decay_hours * math.log(self.config.archival_threshold)

    def predicted_lifespan_days(self, importance: float) -> float:
        """Convenience: lifespan in days."""
        return self.predicted_lifespan_hours(importance) / 24.0

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def decay_samples(self) -> dict[str, float]:
        """Return predicted lifespans for different importance levels (for documentation)."""
        return {
            "high_importance": round(self.predicted_lifespan_days(0.9), 1),
            "medium_importance": round(self.predicted_lifespan_days(0.6), 1),
            "low_importance": round(self.predicted_lifespan_days(0.3), 1),
        }
