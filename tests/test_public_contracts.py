import json
from dataclasses import FrozenInstanceError
from typing import Protocol

import pytest

from memory_agent import (
    EmbeddingPort,
    MemoryRecord,
    MemoryStorePort,
    RetrievalEvidence,
    RetrievalPort,
    SearchResult,
    TrustPolicyPort,
)


@pytest.mark.parametrize(
    "port",
    [MemoryStorePort, EmbeddingPort, RetrievalPort, TrustPolicyPort],
)
def test_public_ports_are_runtime_checkable_protocols(port):
    assert issubclass(port, Protocol)
    assert port._is_runtime_protocol is True



def test_memory_record_legacy_constructor_defaults_remain_compatible():
    memory = MemoryRecord(content="legacy memory")

    assert memory.content == "legacy memory"
    assert memory.memory_type == "episodic"
    assert memory.namespace is None
    assert memory.confidence is None

def test_memory_record_serializes_to_json_safe_dict_with_public_fields():
    memory = MemoryRecord(
        id=7,
        content="Prefers concise answers",
        namespace="user-1",
        confidence=0.9,
        sensitivity="low",
        source="conversation",
        superseded_by="memory-8",
        last_decision_reason="accepted as current preference",
        metadata={"topic": "style"},
        tags=["preference"],
    )

    payload = memory.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["namespace"] == "user-1"
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["metadata"] == {"topic": "style"}


def test_search_result_and_retrieval_evidence_serialize_without_private_objects():
    evidence = RetrievalEvidence(
        score=0.87,
        semantic_score=0.8,
        recency_score=0.7,
        importance_score=0.9,
        strength_score=1.0,
        matched_by=("semantic", "recent", "important"),
        trust="accepted",
        reason="current preference supersedes older preference",
    )
    result = SearchResult(
        memory=MemoryRecord(content="Prefers concise answers"),
        score=0.87,
        semantic_score=0.8,
        recency_score=0.7,
        importance_score=0.9,
        strength_score=1.0,
        evidence=evidence,
    )

    evidence_payload = evidence.to_dict()
    result_payload = result.to_dict()

    assert json.loads(json.dumps(evidence_payload)) == evidence_payload
    assert json.loads(json.dumps(result_payload)) == result_payload
    assert result_payload["memory"]["content"] == "Prefers concise answers"
    assert result_payload["evidence"]["matched_by"] == [
        "semantic",
        "recent",
        "important",
    ]
    assert not any("_" in key and key.startswith("_") for key in result_payload)


def test_retrieval_evidence_is_immutable():
    evidence = RetrievalEvidence(score=0.5)

    with pytest.raises(FrozenInstanceError):
        evidence.score = 0.8
