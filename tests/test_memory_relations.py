"""Public contracts for persistent, typed memory relations.

These tests intentionally define the relation API before its implementation exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord, MemoryRelation


RELATION_TYPES = (
    "related_to",
    "supports",
    "supersedes",
    "contradicts",
    "derived_from",
)


@pytest.fixture
def relation_store(tmp_path: Path) -> tuple[MemoryStore, dict[str, int]]:
    """Create deterministic source/target records in isolated namespaces."""
    store = MemoryStore(tmp_path / "relations.db")
    store.initialize()
    ids = {
        "alpha_source": store.add_memory(
            MemoryRecord(content="alpha source", namespace="alpha"),
            namespace="alpha",
        ),
        "alpha_target": store.add_memory(
            MemoryRecord(content="alpha target", namespace="alpha"),
            namespace="alpha",
        ),
        "beta_target": store.add_memory(
            MemoryRecord(content="beta target", namespace="beta"),
            namespace="beta",
        ),
        "beta_source": store.add_memory(
            MemoryRecord(content="beta source", namespace="beta"),
            namespace="beta",
        ),
    }
    yield store, ids
    store.close()


def _relation(
    source_id: int,
    target_id: int,
    *,
    relation_type: str = "supports",
    namespace: str = "alpha",
    confidence: float = 0.75,
    source: str = "unit-test",
) -> MemoryRelation:
    return MemoryRelation(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        confidence=confidence,
        namespace=namespace,
        source=source,
        is_active=True,
    )


@pytest.mark.parametrize("relation_type", RELATION_TYPES)
def test_memory_relation_serializes_as_json_safe_typed_record(
    relation_type: str,
) -> None:
    relation = _relation(11, 12, relation_type=relation_type)

    payload = relation.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert json.loads(encoded) == payload
    assert payload["source_id"] == 11
    assert payload["target_id"] == 12
    assert payload["relation_type"] == relation_type
    assert payload["confidence"] == 0.75
    assert payload["namespace"] == "alpha"
    assert payload["source"] == "unit-test"
    assert payload["is_active"] is True


def test_relation_persists_with_all_public_fields_after_store_reopen(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "relations.db"
    store = MemoryStore(db_path)
    store.initialize()
    source_id = store.add_memory(MemoryRecord(content="source", namespace="alpha"), namespace="alpha")
    target_id = store.add_memory(MemoryRecord(content="target", namespace="alpha"), namespace="alpha")
    relation_id = store.add_relation(
        _relation(source_id, target_id, relation_type="derived_from", confidence=0.9)
    )
    store.close()

    reopened = MemoryStore(db_path)
    reopened.initialize()
    try:
        relations = reopened.get_relations(
            source_id,
            namespace="alpha",
            active_only=True,
        )
        assert len(relations) == 1
        persisted = relations[0]
        assert persisted.id == relation_id
        assert persisted.source_id == source_id
        assert persisted.target_id == target_id
        assert persisted.relation_type == "derived_from"
        assert persisted.confidence == 0.9
        assert persisted.namespace == "alpha"
        assert persisted.source == "unit-test"
        assert persisted.is_active is True
    finally:
        reopened.close()


def test_duplicate_source_target_type_is_idempotent_and_stored_once(
    relation_store: tuple[MemoryStore, dict[str, int]],
) -> None:
    store, ids = relation_store
    first_id = store.add_relation(
        _relation(ids["alpha_source"], ids["alpha_target"], relation_type="supports")
    )
    second_id = store.add_relation(
        _relation(ids["alpha_source"], ids["alpha_target"], relation_type="supports")
    )

    stored = store.get_relations(
        ids["alpha_source"],
        namespace="alpha",
        active_only=True,
    )
    assert second_id == first_id
    assert len(stored) == 1
    assert stored[0].id == first_id
    assert stored[0].relation_type == "supports"
    assert stored[0].source_id == ids["alpha_source"]
    assert stored[0].target_id == ids["alpha_target"]
    assert stored[0].is_active is True


@pytest.mark.parametrize(
    ("source_ref", "target_ref"),
    [
        ("missing", "missing"),
        ("existing", "missing"),
    ],
    ids=["missing-source-and-target", "missing-target"],
)
def test_relation_rejects_missing_memory_endpoints(
    relation_store: tuple[MemoryStore, dict[str, int]],
    source_ref: str,
    target_ref: str,
) -> None:
    store, ids = relation_store
    source_id = ids["alpha_source"] if source_ref == "existing" else 999_001
    target_id = ids["alpha_target"] if target_ref == "existing" else 999_002

    with pytest.raises(ValueError):
        store.add_relation(_relation(source_id, target_id))


def test_relations_are_namespace_scoped_for_queries_and_targets(
    relation_store: tuple[MemoryStore, dict[str, int]],
) -> None:
    store, ids = relation_store
    store.add_relation(_relation(ids["alpha_source"], ids["alpha_target"]))

    assert store.get_relations(
        ids["alpha_source"],
        namespace="beta",
        active_only=True,
    ) == []
    with pytest.raises(ValueError):
        store.add_relation(
            _relation(
                ids["alpha_source"],
                ids["beta_target"],
                namespace="alpha",
            )
        )
    with pytest.raises(ValueError):
        store.add_relation(
            _relation(
                ids["beta_source"],
                ids["alpha_target"],
                namespace="alpha",
            )
        )



def test_relation_rejects_self_links(
    relation_store: tuple[MemoryStore, dict[str, int]],
) -> None:
    store, ids = relation_store

    with pytest.raises(ValueError):
        store.add_relation(_relation(ids["alpha_source"], ids["alpha_source"]))


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -math.inf])
def test_relation_rejects_nonfinite_confidence(
    relation_store: tuple[MemoryStore, dict[str, int]],
    confidence: float,
) -> None:
    store, ids = relation_store

    with pytest.raises(ValueError):
        store.add_relation(
            _relation(
                ids["alpha_source"],
                ids["alpha_target"],
                confidence=confidence,
            )
        )


def test_relation_rejects_unknown_relation_type(
    relation_store: tuple[MemoryStore, dict[str, int]],
) -> None:
    store, ids = relation_store

    with pytest.raises(ValueError):
        store.add_relation(
            _relation(
                ids["alpha_source"],
                ids["alpha_target"],
                relation_type="untyped_link",
            )
        )


def test_deactivation_removes_active_edge_but_retains_audit_history(
    relation_store: tuple[MemoryStore, dict[str, int]],
) -> None:
    store, ids = relation_store
    relation_id = store.add_relation(
        _relation(
            ids["alpha_source"],
            ids["alpha_target"],
            relation_type="supersedes",
        )
    )

    assert len(
        store.get_relations(
            ids["alpha_source"],
            namespace="alpha",
            active_only=True,
        )
    ) == 1

    store.deactivate_relation(relation_id, namespace="alpha")

    assert store.get_relations(
        ids["alpha_source"],
        namespace="alpha",
        active_only=True,
    ) == []
    history = store.get_relations(
        ids["alpha_source"],
        namespace="alpha",
        active_only=False,
    )
    assert len(history) == 1
    assert history[0].id == relation_id
    assert history[0].relation_type == "supersedes"
    assert history[0].is_active is False
