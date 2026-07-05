"""Tests for the forgetting curve."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from memory_agent.core.config import ForgettingConfig
from memory_agent.core.forgetting import ForgettingCurve
from memory_agent.models import MemoryRecord


@pytest.fixture
def curve() -> ForgettingCurve:
    return ForgettingCurve()


class TestForgettingCurve:
    def test_decay_hours_by_importance(self, curve: ForgettingCurve):
        assert curve.decay_hours_for(0.9) == 720.0     # high
        assert curve.decay_hours_for(0.8) == 720.0     # high boundary
        assert curve.decay_hours_for(0.6) == 168.0     # medium
        assert curve.decay_hours_for(0.5) == 168.0     # medium boundary
        assert curve.decay_hours_for(0.3) == 24.0      # low

    def test_no_decay_when_no_time_passes(self, curve: ForgettingCurve):
        """Strength should remain unchanged when elapsed_hours = 0."""
        result = curve.strength_after_decay(1.0, 0.9, 0)
        assert result == pytest.approx(1.0)

        result = curve.strength_after_decay(0.5, 0.5, 0)
        assert result == pytest.approx(0.5)

    def test_high_importance_decays_slower(self, curve: ForgettingCurve):
        """After 24 hours, a high-importance memory should retain more strength."""
        high = curve.strength_after_decay(1.0, 0.9, 24)
        low = curve.strength_after_decay(1.0, 0.3, 24)
        assert high > low, "High importance should decay slower"

    def test_decay_is_exponential(self, curve: ForgettingCurve):
        """Decay should be exponential (not linear).

        For exponential decay f(t) = e^(-t/D):
        f(a) * f(b) = e^(-(a+b)/D) = f(a+b)
        So s(1) * s(1) should equal s(2).
        """
        s1 = curve.strength_after_decay(1.0, 0.5, 1)
        s2 = curve.strength_after_decay(1.0, 0.5, 2)

        # s1 * s1 should approximate s2
        assert s1 * s1 == pytest.approx(s2, rel=1e-3)

    def test_reinforce_boosts_strength(self, curve: ForgettingCurve):
        mem = MemoryRecord(content="test", strength=0.5)
        new_strength = curve.reinforce(mem)
        assert new_strength == pytest.approx(0.5 + 0.15)  # default boost
        assert mem.access_count == 1
        assert mem.last_accessed_at is not None

    def test_reinforce_caps_at_max(self, curve: ForgettingCurve):
        mem = MemoryRecord(content="test", strength=0.95)
        new_strength = curve.reinforce(mem)
        assert new_strength == pytest.approx(1.0)

    def test_apply_decay_to_memory(self, curve: ForgettingCurve):
        """Should reduce strength for a memory created hours ago."""
        now = datetime.now()
        past = now - timedelta(hours=48)
        mem = MemoryRecord(
            content="test",
            strength=1.0,
            importance=0.5,
            created_at=past.isoformat(),
            last_accessed_at=past.isoformat(),
        )
        new_strength = curve.apply_decay_to_memory(mem, now)
        assert new_strength < 1.0
        assert new_strength > 0.0
        assert mem.strength == new_strength

    def test_fresh_memory_no_decay(self, curve: ForgettingCurve):
        """Memory created now should have no decay."""
        now = datetime.now()
        mem = MemoryRecord(
            content="fresh",
            strength=1.0,
            importance=0.5,
            created_at=now.isoformat(),
            last_accessed_at=now.isoformat(),
        )
        new_strength = curve.apply_decay_to_memory(mem, now)
        assert new_strength == pytest.approx(1.0)

    def test_should_archive_weak_memory(self, curve: ForgettingCurve):
        weak = MemoryRecord(content="weak", strength=0.01)
        strong = MemoryRecord(content="strong", strength=0.5)
        assert curve.should_archive(weak) is True
        assert curve.should_archive(strong) is False

    def test_decay_all(self, curve: ForgettingCurve):
        now = datetime.now()
        past = now - timedelta(hours=24)
        memories = [
            MemoryRecord(
                content="a",
                id=1,
                strength=1.0,
                importance=0.9,
                created_at=past.isoformat(),
                last_accessed_at=past.isoformat(),
            ),
            MemoryRecord(
                content="b",
                id=2,
                strength=1.0,
                importance=0.3,
                created_at=past.isoformat(),
                last_accessed_at=past.isoformat(),
            ),
        ]
        updates = curve.decay_all(memories, now)
        assert len(updates) == 2
        # Low importance should decay more
        assert updates[0][0] > updates[1][0]

    def test_predicted_lifespan(self, curve: ForgettingCurve):
        """High importance should have much longer lifespan."""
        high_days = curve.predicted_lifespan_days(0.9)
        low_days = curve.predicted_lifespan_days(0.3)
        assert high_days > low_days * 10

    def test_decay_samples(self, curve: ForgettingCurve):
        samples = curve.decay_samples()
        assert "high_importance" in samples
        assert "medium_importance" in samples
        assert "low_importance" in samples
        assert samples["high_importance"] > samples["medium_importance"]
