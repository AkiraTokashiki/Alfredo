"""Public contracts for persistent, typed memory relations.

These tests intentionally define the relation API before its implementation exists.
"""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.parametrize(
    ("endpoint", "lifecycle"),
    [
        ("source", "soft-delete"),
        ("target", "soft-delete"),
        ("source", "archive"),
        ("target", "archive"),
        ("source", "hard-delete"),
        ("target", "hard-delete"),
    ],
    ids=lambda value: str(value),
)
def test_inactive_or_deleted_endpoint_has_no_active_relation_and_rejects_new_edge(
    relation_store: tuple[MemoryStore, dict[str, int]],
    endpoint: str,
    lifecycle: str,
) -> None:
    """Endpoint lifecycle changes must invalidate the relation graph edge."""
    store, ids = relation_store
    relation = _relation(ids["alpha_source"], ids["alpha_target"])
    store.add_relation(relation)
    endpoint_id = ids[f"alpha_{endpoint}"]

    if lifecycle == "archive":
        store.archive_memory(endpoint_id, reason="relation lifecycle contract", namespace="alpha")
    else:
        store.delete_memory(
            endpoint_id,
            hard=lifecycle == "hard-delete",
            namespace="alpha",
        )

    assert store.get_relations(
        ids["alpha_source"],
        namespace="alpha",
        active_only=True,
    ) == []
    with pytest.raises(ValueError, match="(active|existing|endpoint)"):
        store.add_relation(relation)


@pytest.mark.parametrize(
    ("source_id", "target_id"),
    [
        ("1", 2),
        (1, "2"),
        (True, 2),
        (1, False),
    ],
    ids=["string-source", "string-target", "bool-source", "bool-target"],
)
def test_relation_rejects_non_integer_endpoint_ids(
    relation_store: tuple[MemoryStore, dict[str, int]],
    source_id: object,
    target_id: object,
) -> None:
    """IDs are database integer identifiers, not coercible strings or booleans."""
    store, _ids = relation_store
    with pytest.raises(TypeError, match="integer|int"):
        store.add_relation(_relation(source_id, target_id))  # type: ignore[arg-type]



@pytest.mark.parametrize(
    "relation_type",
    [1, None, ["supports"]],
    ids=["integer", "none", "list"],
)
def test_relation_rejects_non_string_relation_type(
    relation_store: tuple[MemoryStore, dict[str, int]],
    relation_type: object,
) -> None:
    """Relation type is a named string, never an implicitly coerced value."""
    store, ids = relation_store
    with pytest.raises(TypeError, match="string|str"):
        store.add_relation(
            _relation(
                ids["alpha_source"],
                ids["alpha_target"],
                relation_type=relation_type,  # type: ignore[arg-type]
            )
        )


def test_concurrent_duplicate_add_is_idempotent_under_unique_race(tmp_path: Path) -> None:
    """Two writers racing on one edge must both observe one committed relation."""
    db_path = tmp_path / "relation-race.db"
    seed = MemoryStore(db_path)
    seed.initialize()
    source_id = seed.add_memory(MemoryRecord(content="source", namespace="alpha"), namespace="alpha")
    target_id = seed.add_memory(MemoryRecord(content="target", namespace="alpha"), namespace="alpha")
    seed.close()
    relation = _relation(source_id, target_id)
    barrier = threading.Barrier(2)

    def add_once() -> int:
        writer = MemoryStore(db_path)
        try:
            barrier.wait()
            return writer.add_relation(
                MemoryRelation(
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    namespace=relation.namespace,
                    source=relation.source,
                )
            )
        finally:
            writer.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        relation_ids = list(executor.map(lambda _index: add_once(), range(2)))

    assert relation_ids[0] == relation_ids[1]
    check = MemoryStore(db_path)
    check.initialize()
    try:
        stored = check.get_relations(source_id, namespace="alpha", active_only=True)
        assert len(stored) == 1
        assert stored[0].id == relation_ids[0]
    finally:
        check.close()


def test_memory_relation_is_exported_from_package() -> None:
    """The relation model is part of the package-level public contract."""
    from memory_agent import MemoryRelation as PublicMemoryRelation

    assert PublicMemoryRelation is MemoryRelation
