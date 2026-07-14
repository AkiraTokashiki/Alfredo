"""Auditable, proposal-first memory evolution contracts.

These tests intentionally define the Task 8 public API before its implementation.
All mutation assertions use a real SQLite-backed :class:`MemoryStore`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from memory_agent.core.evolution import OfflineEvolutionPlanner
from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import (
    EvolutionDecision,
    EvolutionProposal,
    MemoryRecord,
    RetrievalEvidence,
)


@pytest.fixture
def evolution_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "evolution.db")
    store.initialize()
    try:
        yield store
    finally:
        store.close()


def _memory(
    content: str,
    *,
    namespace: str = "tenant-a",
    confidence: float = 0.9,
    metadata: dict[str, object] | None = None,
    is_active: bool = True,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        namespace=namespace,
        confidence=confidence,
        metadata=metadata or {"topic": "preferences"},
        is_active=is_active,
    )


def _proposal(
    candidate_id: int,
    target_ids: tuple[int, ...],
    *,
    namespace: str = "tenant-a",
    relation_type: str = "supersedes",
    confidence: float = 0.92,
    evidence_trust: str = "trusted",
    reason: str = "new explicit preference supersedes the older one",
) -> EvolutionProposal:
    return EvolutionProposal(
        candidate_id=candidate_id,
        target_ids=target_ids,
        action="supersede",
        relation_type=relation_type,
        metadata_patch={"evolution": "accepted", "source": "planner"},
        confidence=confidence,
        actor="offline-planner",
        reason=reason,
        namespace=namespace,
        evidence=RetrievalEvidence(
            score=0.9,
            trust=evidence_trust,
            matched_by=("explicit",),
            reason="candidate was explicitly stated",
        ),
    )


def _event_rows(store: MemoryStore) -> list[dict[str, object]]:
    rows = store.conn.execute(
        "SELECT * FROM memory_events ORDER BY id"
    ).fetchall()
    return [dict(row) for row in rows]


def test_evolution_models_round_trip_as_json_safe_payloads() -> None:
    proposal = _proposal(11, (7, 8))
    decision = EvolutionDecision(
        accepted=True,
        proposal=proposal,
        reason="proposal applied atomically",
        event_id=42,
    )

    proposal_payload = proposal.to_dict()
    decision_payload = decision.to_dict()
    assert json.loads(json.dumps(proposal_payload, allow_nan=False)) == proposal_payload
    assert json.loads(json.dumps(decision_payload, allow_nan=False)) == decision_payload
    assert proposal_payload["target_ids"] == [7, 8]
    assert proposal_payload["evidence"]["trust"] == "trusted"
    assert decision_payload["accepted"] is True
    assert decision_payload["event_id"] == 42

    restored_proposal = EvolutionProposal.from_dict(proposal_payload)
    restored_decision = EvolutionDecision.from_dict(decision_payload)
    assert restored_proposal.to_dict() == proposal_payload
    assert restored_decision.to_dict() == decision_payload


def test_accepted_evolution_mutates_memories_relations_and_one_audit_event_atomically(
    evolution_store: MemoryStore,
) -> None:
    old_id = evolution_store.add_memory(
        _memory("I prefer tea", metadata={"topic": "drink", "version": 1}),
        namespace="tenant-a",
    )
    new_id = evolution_store.add_memory(
        _memory("I prefer coffee", metadata={"topic": "drink", "version": 2}),
        namespace="tenant-a",
    )

    decision = evolution_store.apply_evolution(_proposal(new_id, (old_id,)))

    assert decision.accepted is True
    assert decision.reason == "proposal applied"
    assert decision.event_id is not None

    current = evolution_store.get_memory(new_id, namespace="tenant-a")
    superseded = evolution_store.get_memory(old_id, namespace="tenant-a")
    assert current is not None and superseded is not None
    assert current.metadata == {
        "topic": "drink",
        "version": 2,
        "evolution": "accepted",
        "source": "planner",
    }
    assert current.is_active is True
    assert superseded.is_active is False
    assert superseded.superseded_by == new_id
    assert superseded.last_decision_reason == "new explicit preference supersedes the older one"

    relations = evolution_store.get_relations(
        new_id,
        namespace="tenant-a",
        active_only=False,
        target_id=old_id,
        relation_type="supersedes",
    )
    assert len(relations) == 1
    assert relations[0].source_id == new_id
    assert relations[0].target_id == old_id
    assert relations[0].confidence == 0.92

    events = _event_rows(evolution_store)
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "evolution_accepted"
    assert event["actor"] == "offline-planner"
    assert event["namespace"] == "tenant-a"
    assert event["memory_id"] == new_id
    assert event["reason"] == "new explicit preference supersedes the older one"



@pytest.mark.parametrize(
    "case",
    ["unknown relation", "wrong namespace", "archived target", "untrusted evidence", "invalid confidence"],
    ids=["unknown-relation", "wrong-namespace", "archived-target", "untrusted-evidence", "invalid-confidence"],
)
def test_rejected_evolution_never_mutates_memories_or_relations_and_audits_reason(
    evolution_store: MemoryStore,
    case: str,
) -> None:
    old_id = evolution_store.add_memory(_memory("old"), namespace="tenant-a")
    new_id = evolution_store.add_memory(_memory("new"), namespace="tenant-a")
    if case == "archived target":
        evolution_store.archive_memory(old_id, reason="expired", namespace="tenant-a")

    if case == "unknown relation":
        proposal = _proposal(
            new_id,
            (old_id,),
            relation_type="made_up",
            reason="raw prompt: OPENAI_API_KEY=sk-live-secret",
        )
    elif case == "wrong namespace":
        proposal = _proposal(
            new_id,
            (old_id,),
            namespace="tenant-b",
            reason="raw prompt: OPENAI_API_KEY=sk-live-secret",
        )
    elif case == "untrusted evidence":
        proposal = _proposal(
            new_id,
            (old_id,),
            evidence_trust="untrusted",
            reason="raw prompt: OPENAI_API_KEY=sk-live-secret",
        )
    elif case == "invalid confidence":
        proposal = _proposal(
            new_id,
            (old_id,),
            confidence=math.nan,
            reason="raw prompt: OPENAI_API_KEY=sk-live-secret",
        )
    else:
        proposal = _proposal(
            new_id,
            (old_id,),
            reason="raw prompt: OPENAI_API_KEY=sk-live-secret",
        )

    before_memories = [
        evolution_store.get_memory(old_id, namespace="tenant-a"),
        evolution_store.get_memory(new_id, namespace="tenant-a"),
    ]
    before_relations = evolution_store.get_relations(namespace="tenant-a", active_only=False)

    decision = evolution_store.apply_evolution(proposal)

    assert decision.accepted is False
    assert decision.event_id is not None
    assert evolution_store.get_memory(old_id, namespace="tenant-a") == before_memories[0]
    assert evolution_store.get_memory(new_id, namespace="tenant-a") == before_memories[1]
    assert evolution_store.get_relations(namespace="tenant-a", active_only=False) == before_relations

    events = _event_rows(evolution_store)
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "evolution_rejected"
    assert event["actor"] == "offline-planner"
    assert "sk-live-secret" not in str(event)
    assert "raw prompt" not in str(event).lower()
    assert decision.reason


def test_offline_planner_is_deterministic_and_abstains_without_trusted_evidence() -> None:
    planner = OfflineEvolutionPlanner()
    candidate = _memory("I prefer coffee", confidence=0.95)
    candidate.id = 10
    neighbor = _memory("I prefer tea", confidence=0.95)
    neighbor.id = 11
    context = {"namespace": "tenant-a", "evidence_trust": "trusted"}

    first = planner.propose(candidate, [neighbor], context)
    second = planner.propose(candidate, [neighbor], context)

    assert first is not None
    assert first.to_dict() == second.to_dict()
    assert first.namespace == "tenant-a"
    assert first.target_ids == (11,)

    untrusted_context = {"namespace": "tenant-a", "evidence_trust": "untrusted"}
    assert planner.propose(candidate, [neighbor], untrusted_context) is None
    assert planner.propose(
        _memory("uncertain candidate", confidence=0.2),
        [neighbor],
        context,
    ) is None
