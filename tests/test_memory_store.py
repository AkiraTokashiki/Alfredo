"""Tests for MemoryStore."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


@pytest.fixture
def store() -> MemoryStore:
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    s = MemoryStore(db.name)
    s.initialize()
    yield s
    s.close()
    Path(db.name).unlink(missing_ok=True)


class TestMemoryStore:
    def test_initialize_creates_tables(self, store: MemoryStore):
        """Tables should exist after initialize()."""
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in tables]
        assert "memories" in names
        assert "embeddings" in names
        assert "memory_tags" in names
        assert "sessions" in names
        assert "session_memories" in names

    def test_add_and_get_memory(self, store: MemoryStore):
        mem = MemoryRecord(
            content="El usuario prefiere respuestas en espanol",
            memory_type="preference",
            importance=0.9,
            tags=["language", "spanish"],
        )
        mem_id = store.add_memory(mem)
        assert mem_id > 0
        assert mem.id == mem_id

        fetched = store.get_memory(mem_id)
        assert fetched is not None
        assert fetched.content == mem.content
        assert fetched.memory_type == "preference"
        assert fetched.importance == 0.9
        assert "spanish" in fetched.tags

    def test_update_memory(self, store: MemoryStore):
        mem = MemoryRecord(content="test", importance=0.5)
        mem_id = store.add_memory(mem)
        mem.importance = 0.95
        store.update_memory(mem)
        fetched = store.get_memory(mem_id)
        assert fetched is not None
        assert fetched.importance == 0.95

    def test_soft_delete(self, store: MemoryStore):
        mem = MemoryRecord(content="delete me")
        mem_id = store.add_memory(mem)
        store.delete_memory(mem_id, hard=False)
        fetched = store.get_memory(mem_id)
        assert fetched is not None
        assert fetched.is_active is False

    def test_hard_delete(self, store: MemoryStore):
        mem = MemoryRecord(content="hard delete me")
        mem_id = store.add_memory(mem)
        store.delete_memory(mem_id, hard=True)
        fetched = store.get_memory(mem_id)
        assert fetched is None

    def test_count_memories(self, store: MemoryStore):
        assert store.count_memories() == 0
        store.add_memory(MemoryRecord(content="a"))
        store.add_memory(MemoryRecord(content="b"))
        assert store.count_memories() == 2

    def test_get_all_active_memories(self, store: MemoryStore):
        m1 = store.add_memory(MemoryRecord(content="active", importance=0.5))
        m2 = store.add_memory(MemoryRecord(content="important", importance=0.9))
        store.add_memory(MemoryRecord(content="hidden", importance=0.1))
        store.delete_memory(3)  # soft-delete

        all_mem = store.get_all_active_memories()
        assert len(all_mem) == 2
        # Should be sorted by importance descending
        assert all_mem[0].importance == 0.9

    def test_memories_by_type(self, store: MemoryStore):
        store.add_memory(MemoryRecord(content="ep", memory_type="episodic"))
        store.add_memory(MemoryRecord(content="sem", memory_type="semantic"))
        store.add_memory(MemoryRecord(content="pref", memory_type="preference"))

        eps = store.get_memories_by_type("episodic")
        assert len(eps) == 1
        assert eps[0].content == "ep"

        prefs = store.get_memories_by_type("preference")
        assert len(prefs) == 1
        assert prefs[0].content == "pref"

    def test_memories_by_tag(self, store: MemoryStore):
        store.add_memory(MemoryRecord(content="a", tags=["foo"]))
        store.add_memory(MemoryRecord(content="b", tags=["bar"]))
        store.add_memory(MemoryRecord(content="c", tags=["foo", "baz"]))

        foos = store.get_memories_by_tag("foo")
        assert len(foos) == 2
        bars = store.get_memories_by_tag("bar")
        assert len(bars) == 1

    def test_sessions(self, store: MemoryStore):
        sid = store.create_session("test session")
        assert sid > 0

        m1 = store.add_memory(MemoryRecord(content="mem1"))
        m2 = store.add_memory(MemoryRecord(content="mem2"))

        store.link_memory_to_session(sid, m1, turn_index=1)
        store.link_memory_to_session(sid, m2, turn_index=2)

        mems = store.get_session_memories(sid)
        assert len(mems) == 2
        assert mems[0].content == "mem1"

        store.end_session(sid)
        sessions = store.get_recent_sessions()
        assert len(sessions) >= 1
        assert sessions[0].ended_at is not None

    def test_embeddings(self, store: MemoryStore):
        mid = store.add_memory(MemoryRecord(content="test embedding"))
        store.save_embedding(mid, b"vector_data_here", "test-model")

        result = store.get_embedding(mid)
        assert result is not None
        assert result[0] == b"vector_data_here"
        assert result[1] == "test-model"

        all_emb = store.get_all_embeddings()
        assert len(all_emb) == 1
        assert all_emb[0][0] == mid

    def test_keyword_search(self, store: MemoryStore):
        store.add_memory(MemoryRecord(content="el gato esta en la casa"))
        store.add_memory(MemoryRecord(content="el perro esta en el parque"))
        store.add_memory(MemoryRecord(content="me gusta el cafe"))

        results = store.search_keywords("gato")
        assert len(results) == 1
        assert "gato" in results[0].content

        results = store.search_keywords("esta")
        assert len(results) == 2

    def test_archive_below_threshold(self, store: MemoryStore):
        store.add_memory(MemoryRecord(content="strong", strength=0.5))
        store.add_memory(MemoryRecord(content="weak", strength=0.01))

        archived = store.archive_below_threshold(0.1)
        assert archived == 1

        active = store.get_all_active_memories()
        assert len(active) == 1
        assert active[0].content == "strong"

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        s = MemoryStore(db_path)
        s.initialize()
        s.add_memory(MemoryRecord(content="ctx test"))
        assert s.count_memories() == 1
        s.close()
        Path(db_path).unlink(missing_ok=True)
