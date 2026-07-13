"""Contract tests for the MCP adapter facade boundary.

These tests intentionally exercise the public ``MemoryAgent`` facade only.  The
fake raises if an adapter reaches into retrieval, storage, or forgetting
implementation objects; this keeps the MCP boundary honest while the facade
implementation is being completed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.integrations import mcp_server
from memory_agent.models import MemoryRecord, RetrievalEvidence


@dataclass
class _Call:
    name: str
    kwargs: dict


class _FacadeSpy:
    """Minimal public MemoryAgent facade used by the adapter tests."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        # _ensure_session only needs to see an already-active session.
        self.state = SimpleNamespace(session_id=41)

    @property
    def retrieval(self):  # pragma: no cover - reached only on an invalid adapter
        raise AssertionError("MCP adapter bypassed MemoryAgent.search_memories")

    @property
    def store(self):  # pragma: no cover - reached only on an invalid adapter
        raise AssertionError("MCP adapter bypassed MemoryAgent facade for storage")

    @property
    def forgetting(self):  # pragma: no cover - reached only on an invalid adapter
        raise AssertionError("MCP adapter bypassed MemoryAgent facade for forgetting")

    def perceive(
        self,
        user_input: str,
        agent_response: str | None = None,
        *,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        self.calls.append(
            _Call(
                "perceive",
                {
                    "user_input": user_input,
                    "agent_response": agent_response,
                    "namespace": namespace,
                    "user_id": user_id,
                },
            )
        )
        return {
            "namespace": namespace,
            "recollections": [
                {
                    "id": 11,
                    "content": "The user prefers Python.",
                    "type": "preference",
                }
            ],
            "new_memories": [],
            "evidence": [
                {"id": 11, "trust": "trusted", "reason": "current preference"},
                {
                    "id": 22,
                    "trust": "untrusted",
                    "reason": "omitted: low confidence",
                },
            ],
            "selected_ids": [11],
            "dropped_ids": [22],
            "lifecycle": {"status": "recalled", "namespace": namespace},
        }

    def search_memories(
        self,
        query: str,
        *,
        top_k: int = 5,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> dict:
        self.calls.append(
            _Call(
                "search_memories",
                {
                    "query": query,
                    "top_k": top_k,
                    "memory_type": memory_type,
                    "namespace": namespace,
                },
            )
        )
        return {
            "namespace": namespace,
            "results": [
                {
                    "id": 11,
                    "content": "The user prefers Python.",
                    "type": "preference",
                    "score": 0.91,
                }
            ],
            "evidence": [
                {"id": 11, "trust": "trusted", "reason": "semantic match"},
                {"id": 22, "trust": "untrusted", "reason": "dropped: stale"},
            ],
            "selected_ids": [11],
            "dropped_ids": [22],
            "lifecycle": {"status": "searched", "namespace": namespace},
        }

    def store_memory(self, memory, *, namespace: str | None = None) -> int:
        self.calls.append(
            _Call(
                "store_memory",
                {
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "importance": memory.importance,
                    "tags": list(memory.tags),
                    "namespace": namespace,
                },
            )
        )
        return 77


@pytest.fixture
def facade(monkeypatch: pytest.MonkeyPatch) -> _FacadeSpy:
    fake = _FacadeSpy()
    monkeypatch.setattr(mcp_server, "_get_agent", lambda: fake)
    return fake


def test_memory_perceive_forwards_namespace_to_facade_and_returns_evidence_safe_json(
    facade: _FacadeSpy,
) -> None:
    raw = asyncio.run(
        mcp_server.memory_perceive(
            "What language do I prefer?", namespace="tenant-a", top_k=3
        )
    )

    payload = json.loads(raw)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    assert facade.calls == [
        _Call(
            "perceive",
            {
                "user_input": "What language do I prefer?",
                "agent_response": None,
                "namespace": "tenant-a",
                "user_id": None,
            },
        )
    ]
    assert payload["namespace"] == "tenant-a"
    assert payload["selected_ids"] == [11]
    assert payload["dropped_ids"] == [22]
    assert payload["evidence"] == [
        {"id": 11, "trust": "trusted", "reason": "current preference"},
        {"id": 22, "trust": "untrusted", "reason": "omitted: low confidence"},
    ]
    assert payload["lifecycle"]["status"] == "recalled"


def test_memory_search_uses_facade_and_preserves_namespace_selection_and_trust(
    facade: _FacadeSpy,
) -> None:
    raw = asyncio.run(
        mcp_server.memory_search(
            "preferred language",
            namespace="tenant-a",
            top_k=1,
            memory_type="preference",
        )
    )

    payload = json.loads(raw)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    assert facade.calls == [
        _Call(
            "search_memories",
            {
                "query": "preferred language",
                "top_k": 1,
                "memory_type": "preference",
                "namespace": "tenant-a",
            },
        )
    ]
    assert payload["namespace"] == "tenant-a"
    assert payload["selected_ids"] == [11]
    assert payload["dropped_ids"] == [22]
    assert payload["evidence"][0]["trust"] == "trusted"
    assert payload["evidence"][0]["reason"] == "semantic match"
    assert payload["evidence"][1]["trust"] == "untrusted"
    assert payload["lifecycle"] == {"status": "searched", "namespace": "tenant-a"}


def test_memory_store_forwards_namespace_through_public_facade(
    facade: _FacadeSpy,
) -> None:
    raw = asyncio.run(
        mcp_server.memory_store(
            "The user prefers Python.",
            namespace="tenant-a",
            memory_type="preference",
            importance=0.8,
            tags="language, durable",
        )
    )

    payload = json.loads(raw)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    assert facade.calls == [
        _Call(
            "store_memory",
            {
                "content": "The user prefers Python.",
                "memory_type": "preference",
                "importance": 0.8,
                "tags": ["language", "durable"],
                "namespace": "tenant-a",
            },
        )
    ]
    assert payload["id"] == 77
    assert payload["namespace"] == "tenant-a"
    assert payload["status"] == "stored"


@pytest.fixture
def direct_agent(tmp_path: Path):
    config = MemoryAgentConfig.default()
    config.db_path = str(tmp_path / "memory.db")
    config.embedding.provider = "deterministic"
    config.embedding.dimension = 32
    config.retrieval.min_score = 0.0
    config.retrieval.top_k = 4
    config.retrieval.candidate_k = 4
    config.retrieval.context_budget_chars = 100_000
    config.retrieval.reserved_context_chars = 0

    agent = MemoryAgent(config=config)
    try:
        yield agent
    finally:
        agent.close()


def _seed_direct_facade_memories(agent: MemoryAgent) -> dict[str, int]:
    ids: dict[str, int] = {}
    for namespace in ("tenant-a", "tenant-b"):
        for trust, confidence in (("trusted", 0.9), ("untrusted", 0.1)):
            key = f"{namespace}-{trust}"
            ids[key] = agent.store_memory(
                MemoryRecord(
                    content=f"{namespace} {trust} Python preference",
                    memory_type="semantic",
                    importance=0.9,
                    confidence=confidence,
                ),
                namespace=namespace,
            )
    return ids


def test_memory_agent_search_applies_trust_and_namespace_policy(
    direct_agent: MemoryAgent,
) -> None:
    ids = _seed_direct_facade_memories(direct_agent)

    payload = direct_agent.search_memories(
        "Python preference", top_k=4, namespace="tenant-a"
    )
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    trusted_id = ids["tenant-a-trusted"]
    untrusted_id = ids["tenant-a-untrusted"]
    other_namespace_ids = {ids["tenant-b-trusted"], ids["tenant-b-untrusted"]}
    assert payload["namespace"] == "tenant-a"
    assert set(payload["selected_ids"]) == {trusted_id}
    assert set(payload["dropped_ids"]) == {untrusted_id}
    assert not other_namespace_ids.intersection(
        payload["selected_ids"] + payload["dropped_ids"]
    )
    assert {item["memory"]["id"] for item in payload["results"]} == {trusted_id}

    evidence = {item["id"]: item for item in payload["evidence"]}
    assert evidence[trusted_id]["trust"] == "trusted"
    assert evidence[trusted_id]["reason"]
    assert evidence[untrusted_id]["trust"] == "untrusted"
    assert "untrusted" in evidence[untrusted_id]["reason"]
    assert payload["lifecycle"] == {
        "operation": "search",
        "status": "searched",
        "namespace": "tenant-a",
    }


def test_memory_agent_forget_respects_namespace_without_cross_archiving(
    direct_agent: MemoryAgent,
) -> None:
    ids = _seed_direct_facade_memories(direct_agent)
    trusted_a = ids["tenant-a-trusted"]

    wrong_namespace = direct_agent.forget_memory(trusted_a, namespace="tenant-b")

    assert wrong_namespace == {
        "id": trusted_a,
        "namespace": "tenant-b",
        "status": "not_found",
        "trust": "unknown",
        "reason": "memory not found in namespace",
        "lifecycle": "unchanged",
    }
    assert direct_agent.store.get_memory(trusted_a, namespace="tenant-a").is_active

    archived = direct_agent.forget_memory(trusted_a, namespace="tenant-a")
    assert archived["status"] == "archived"
    assert archived["lifecycle"] == "archived"
    assert not direct_agent.store.get_memory(
        trusted_a, namespace="tenant-a"
    ).is_active
    assert direct_agent.store.get_memory(
        ids["tenant-b-trusted"], namespace="tenant-b"
    ).is_active
def test_memory_agent_forget_uses_configured_trust_policy(
    direct_agent: MemoryAgent,
) -> None:
    ids = _seed_direct_facade_memories(direct_agent)

    class _StrictPolicy:
        def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
            return RetrievalEvidence(trust="policy-trusted", reason="strict policy")

    direct_agent.trust_policy = _StrictPolicy()
    result = direct_agent.forget_memory(ids["tenant-a-trusted"], namespace="tenant-a")

    assert result["trust"] == "policy-trusted"
    assert "strict policy" in result["reason"]
def test_mcp_perceive_keeps_full_packet_and_store_reports_effective_namespace(
    facade: _FacadeSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    facade.namespace = "tenant-a"

    def perceive(*args, **kwargs):
        selected = [{"id": 11}, {"id": 12}]
        omitted = [{"id": 22}]
        packet = SimpleNamespace(
            selected_ids=[11, 12],
            dropped_ids=[22],
            selected=selected,
            omitted=omitted,
            reasons={},
        )
        return {
            "recollections": selected,
            "new_memories": [],
            "recall_packet": packet,
            "lifecycle": {"namespace": "tenant-a", "status": "recalled"},
        }

    def store(memory, *, namespace=None):
        memory.namespace = facade.namespace
        return 99

    facade.perceive = perceive
    facade.store_memory = store
    monkeypatch.setattr(mcp_server, "_get_agent", lambda: facade)

    perceived = json.loads(
        asyncio.run(mcp_server.memory_perceive("query", top_k=1))
    )
    stored = json.loads(
        asyncio.run(mcp_server.memory_store("fact", namespace=None))
    )

    assert perceived["namespace"] == "tenant-a"
    assert perceived["selected_ids"] == [11, 12]
    assert perceived["dropped_ids"] == [22]
    assert stored["namespace"] == "tenant-a"
    assert stored["lifecycle"]["namespace"] == "tenant-a"

    def store_empty(memory, *, namespace=None):
        memory.namespace = ""
        return 100

    facade.store_memory = store_empty
    stored_empty = json.loads(
        asyncio.run(mcp_server.memory_store("fact", namespace=""))
    )
    assert stored_empty["namespace"] == ""
    assert stored_empty["lifecycle"]["namespace"] == ""
def test_memory_agent_store_uses_active_scope_over_memory_namespace(
    direct_agent: MemoryAgent,
) -> None:
    direct_agent.namespace = "tenant-a"
    memory = MemoryRecord(
        content="must stay in active tenant",
        memory_type="semantic",
        namespace="tenant-b",
        confidence=0.9,
    )

    memory_id = direct_agent.store_memory(memory)

    assert memory.namespace == "tenant-a"
    assert direct_agent.store.get_memory(memory_id, namespace="tenant-a") is not None
    assert direct_agent.store.get_memory(memory_id, namespace="tenant-b") is None
