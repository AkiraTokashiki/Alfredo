"""Namespace isolation and schema migration tests for MemoryStore."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


@pytest.fixture
def db_path() -> Path:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory) / "memory.db"


def test_same_content_in_namespaces_never_crosses_queries(db_path: Path):
    store = MemoryStore(db_path)
    store.initialize()

    memory_a = MemoryRecord(
        content="the same private fact",
        namespace="tenant-a",
        tags=["shared-tag"],
        strength=0.01,
    )
    memory_b = MemoryRecord(
        content="the same private fact",
        namespace="tenant-b",
        tags=["shared-tag"],
        strength=0.01,
    )
    id_a = store.add_memory(memory_a)
    id_b = store.add_memory(memory_b)
    session_a = store.create_session("A", namespace="tenant-a")
    session_b = store.create_session("B", namespace="tenant-b")
    store.link_memory_to_session(session_a, id_a, namespace="tenant-a")
    store.link_memory_to_session(session_b, id_b, namespace="tenant-b")
    store.save_embedding(id_a, b"a", "test", namespace="tenant-a")
    store.save_embedding(id_b, b"b", "test", namespace="tenant-b")

    assert [m.id for m in store.get_all_active_memories(namespace="tenant-a")] == [id_a]
    assert [m.id for m in store.get_all_active_memories(namespace="tenant-b")] == [id_b]
    assert store.count_memories(namespace="tenant-a") == 1
    assert store.count_memories(namespace="tenant-b") == 1
    assert [m.id for m in store.search_keywords("private", namespace="tenant-a")] == [id_a]
    assert [m.id for m in store.get_memories_by_tag("shared-tag", namespace="tenant-b")] == [id_b]
    assert [m.id for m in store.get_session_memories(session_a, namespace="tenant-a")] == [id_a]
    assert store.get_session_memories(session_a, namespace="tenant-b") == []
    assert [s.id for s in store.get_recent_sessions(namespace="tenant-a")] == [session_a]
    assert store.get_embedding(id_a, namespace="tenant-b") is None
    assert store.get_all_embeddings(namespace="tenant-a") == [(id_a, b"a")]

    # Namespace predicates must protect writes, not only reads.
    memory_a.confidence = 0.88
    store.update_memory(memory_a, namespace="tenant-b")
    assert store.get_memory(id_a, namespace="tenant-a").confidence is None
    store.archive_memory(id_a, namespace="tenant-b", reason="wrong namespace")
    assert store.get_memory(id_a, namespace="tenant-a").is_active is True
    store.delete_memory(id_a, namespace="tenant-b")
    assert store.get_memory(id_a, namespace="tenant-a").is_active is True

    assert store.archive_below_threshold(0.1, namespace="tenant-a") == 1
    assert store.get_memory(id_a, namespace="tenant-a").is_active is False
    assert store.get_memory(id_b, namespace="tenant-b").is_active is True
    store.close()


def test_initialize_migrates_existing_database_idempotently(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'episodic',
            importance REAL NOT NULL DEFAULT 0.5,
            strength REAL NOT NULL DEFAULT 1.0,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            metadata TEXT DEFAULT '{}',
            is_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE embeddings (
            memory_id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            model_name TEXT NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );
        CREATE TABLE memory_tags (
            memory_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (memory_id, tag),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT
        );
        CREATE TABLE session_memories (
            session_id INTEGER NOT NULL,
            memory_id INTEGER NOT NULL,
            turn_index INTEGER,
            PRIMARY KEY (session_id, memory_id)
        );
        INSERT INTO memories (content, metadata) VALUES ('legacy fact', '{"legacy": true}');
        """
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    store.initialize()
    store.initialize()

    columns = [row[1] for row in store.conn.execute("PRAGMA table_info(memories)")]
    assert columns.count("namespace") == 1
    for column in (
        "confidence",
        "sensitivity",
        "source",
        "superseded_by",
        "last_decision_reason",
    ):
        assert columns.count(column) == 1
    legacy = store.get_memory(1)
    assert legacy is not None
    assert legacy.content == "legacy fact"
    assert legacy.namespace is None
    assert legacy.confidence is None
    store.close()


def test_update_without_namespace_is_legacy_and_explicit_preserves_owner(
    db_path: Path,
):
    store = MemoryStore(db_path)
    store.initialize()
    memory_id = store.add_memory(MemoryRecord(content="before", namespace="tenant-a"))

    legacy_update = MemoryRecord(id=memory_id, content="legacy update")
    store.update_memory(legacy_update)
    fetched = store.get_memory(memory_id, namespace="tenant-a")
    assert fetched is not None
    assert fetched.content == "legacy update"
    assert fetched.namespace == "tenant-a"

    explicit_update = MemoryRecord(id=memory_id, content="explicit update")
    store.update_memory(explicit_update, namespace="tenant-a")
    fetched = store.get_memory(memory_id, namespace="tenant-a")
    assert fetched is not None
    assert fetched.content == "explicit update"
    assert fetched.namespace == "tenant-a"
    conflicting_update = MemoryRecord(
        id=memory_id,
        content="scope wins",
        namespace="tenant-b",
    )
    store.update_memory(conflicting_update, namespace="tenant-a")
    fetched = store.get_memory(memory_id, namespace="tenant-a")
    assert fetched is not None
    assert fetched.content == "scope wins"
    assert fetched.namespace == "tenant-a"
    assert store.get_memory(memory_id, namespace="tenant-b") is None
    implicit_conflicting_update = MemoryRecord(
        id=memory_id,
        content="owner remains",
        namespace="tenant-b",
    )
    store.update_memory(implicit_conflicting_update)
    fetched = store.get_memory(memory_id, namespace="tenant-a")
    assert fetched is not None
    assert fetched.content == "owner remains"
    assert fetched.namespace == "tenant-a"
    assert store.get_memory(memory_id, namespace="tenant-b") is None
    store.close()


def test_update_without_scope_cannot_reassign_namespace(db_path: Path):
    store = MemoryStore(db_path)
    store.initialize()
    memory_id = store.add_memory(MemoryRecord(content="owned", namespace="tenant-a"))

    conflicting = MemoryRecord(
        id=memory_id,
        content="still owned",
        namespace="tenant-b",
    )
    store.update_memory(conflicting)

    fetched = store.get_memory(memory_id, namespace="tenant-a")
    assert fetched is not None
    assert fetched.content == "still owned"
    assert fetched.namespace == "tenant-a"
    assert store.get_memory(memory_id, namespace="tenant-b") is None
    store.close()


def test_recent_sessions_do_not_report_cross_namespace_memory_links(
    db_path: Path,
):
    store = MemoryStore(db_path)
    store.initialize()
    memory_a = store.add_memory(MemoryRecord(content="A", namespace="tenant-a"))
    memory_b = store.add_memory(MemoryRecord(content="B", namespace="tenant-b"))
    session_a = store.create_session("A", namespace="tenant-a")
    store.link_memory_to_session(session_a, memory_a, namespace="tenant-a")
    store.conn.execute(
        "INSERT INTO session_memories (session_id, memory_id, turn_index) "
        "VALUES (?, ?, ?)",
        (session_a, memory_b, 2),
    )
    store.conn.commit()

    sessions = store.get_recent_sessions(namespace="tenant-a")
    assert len(sessions) == 1
    assert sessions[0].memory_ids == [memory_a]
    store.close()
