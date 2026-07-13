"""SQLite-backed persistent memory storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_agent.models import MemoryRecord, SessionRecord


class MemoryStore:
    """Persistent memory storage backed by SQLite.

    Manages memories, embeddings, tags, sessions, and the
    many-to-many relationships between them.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS memories (
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
        is_active INTEGER NOT NULL DEFAULT 1,
        namespace TEXT,
        confidence REAL,
        sensitivity TEXT,
        source TEXT,
        superseded_by INTEGER,
        last_decision_reason TEXT
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        memory_id INTEGER PRIMARY KEY,
        vector BLOB NOT NULL,
        model_name TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS memory_tags (
        memory_id INTEGER NOT NULL,
        tag TEXT NOT NULL,
        PRIMARY KEY (memory_id, tag),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at TEXT,
        namespace TEXT
    );

    CREATE TABLE IF NOT EXISTS session_memories (
        session_id INTEGER NOT NULL,
        memory_id INTEGER NOT NULL,
        turn_index INTEGER,
        PRIMARY KEY (session_id, memory_id),
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at);
    CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active) WHERE is_active = 1;
    CREATE INDEX IF NOT EXISTS idx_session_memories_session ON session_memories(session_id);
    """

    SCHEMA_VERSION = 2

    def initialize(self) -> None:
        """Create tables and apply idempotent schema migrations."""
        self.conn.executescript(self.SCHEMA_SQL)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        columns_by_table = {
            "memories": {
                "namespace": "TEXT",
                "confidence": "REAL",
                "sensitivity": "TEXT",
                "source": "TEXT",
                "superseded_by": "INTEGER",
                "last_decision_reason": "TEXT",
            },
            "sessions": {"namespace": "TEXT"},
        }
        for table, columns in columns_by_table.items():
            existing = {
                row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")
            }
            for column, definition in columns.items():
                if column not in existing:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                    )
        self.conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(self.SCHEMA_VERSION),),
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
        )
        self.conn.commit()
    def commit(self) -> None:
        """Commit pending lifecycle mutations."""
        self.conn.commit()
    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def add_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        """Insert a new memory record. Returns the new row id."""
        if namespace is not None:
            memory.namespace = namespace
        now = datetime.now().isoformat()
        cur = self.conn.execute(
            """INSERT INTO memories
               (content, memory_type, importance, strength,
                access_count, last_accessed_at, created_at, updated_at,
                metadata, is_active, namespace, confidence, sensitivity,
                source, superseded_by, last_decision_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.content,
                memory.memory_type,
                memory.importance,
                memory.strength,
                memory.access_count,
                memory.last_accessed_at or now,
                memory.created_at or now,
                now,
                json.dumps(memory.metadata),
                1 if memory.is_active else 0,
                memory.namespace,
                memory.confidence,
                memory.sensitivity,
                memory.source,
                memory.superseded_by,
                memory.last_decision_reason,
            ),
        )
        memory_id = cur.lastrowid
        memory.id = memory_id

        for tag in memory.tags:
            self.conn.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                (memory_id, tag),
            )

        if commit:
            self.conn.commit()
        return memory_id

    def get_memory(
        self, memory_id: int, *, namespace: str | None = None
    ) -> MemoryRecord | None:
        """Fetch a single memory, optionally constrained to a namespace."""
        query = "SELECT * FROM memories WHERE id = ?"
        params: list[Any] = [memory_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def update_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Update an existing memory, optionally constrained to a namespace."""
        now = datetime.now().isoformat()
        stored_namespace = namespace
        if namespace is None and memory.id is not None:
            existing = self.conn.execute(
                "SELECT namespace FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()
            if existing is not None:
                stored_namespace = existing["namespace"]
        query = """UPDATE memories SET
               content = ?, memory_type = ?, importance = ?, strength = ?,
               access_count = ?, last_accessed_at = ?, updated_at = ?,
               metadata = ?, is_active = ?, namespace = ?, confidence = ?,
               sensitivity = ?, source = ?, superseded_by = ?,
               last_decision_reason = ?
               WHERE id = ?"""
        params: list[Any] = [
            memory.content,
            memory.memory_type,
            memory.importance,
            memory.strength,
            memory.access_count,
            memory.last_accessed_at or now,
            now,
            json.dumps(memory.metadata),
            1 if memory.is_active else 0,
            stored_namespace,
            memory.confidence,
            memory.sensitivity,
            memory.source,
            memory.superseded_by,
            memory.last_decision_reason,
            memory.id,
        ]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        cur = self.conn.execute(query, params)
        if cur.rowcount and memory.id is not None:
            self.conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory.id,))
            for tag in memory.tags:
                self.conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (memory.id, tag),
                )
        if commit:
            self.conn.commit()

    def delete_memory(
        self,
        memory_id: int,
        *,
        hard: bool = False,
        namespace: str | None = None,
    ) -> None:
        """Soft-delete or hard-delete a memory in an optional namespace."""
        predicate = "id = ?"
        params: list[Any] = [memory_id]
        if namespace is not None:
            predicate += " AND namespace = ?"
            params.append(namespace)
        if hard:
            self.conn.execute(f"DELETE FROM memories WHERE {predicate}", params)
        else:
            self.conn.execute(
                "UPDATE memories SET is_active = 0, updated_at = datetime('now') "
                f"WHERE {predicate}",
                params,
            )
        self.conn.commit()

    def archive_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Archive one memory and preserve the reason in metadata."""
        memory = self.get_memory(memory_id, namespace=namespace)
        if memory is None:
            return
        merged_metadata = dict(memory.metadata)
        merged_metadata["archival_reason"] = reason
        if metadata:
            merged_metadata.update(metadata)
        memory.is_active = False
        memory.metadata = merged_metadata
        self.update_memory(memory, namespace=namespace, commit=commit)

    def count_memories(
        self, *, active_only: bool = True, namespace: str | None = None
    ) -> int:
        """Total number of memories in the store."""
        query = "SELECT COUNT(*) FROM memories"
        predicates: list[str] = []
        params: list[Any] = []
        if active_only:
            predicates.append("is_active = 1")
        if namespace is not None:
            predicates.append("namespace = ?")
            params.append(namespace)
        if predicates:
            query += " WHERE " + " AND ".join(predicates)
        row = self.conn.execute(query, params).fetchone()
        return row[0]


    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def get_all_active_memories(
        self, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Return all active memories, optionally constrained to a namespace."""
        query = "SELECT * FROM memories WHERE is_active = 1"
        params: list[Any] = []
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY importance DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_type(
        self, memory_type: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Filter active memories by type and optional namespace."""
        query = "SELECT * FROM memories WHERE memory_type = ? AND is_active = 1"
        params: list[Any] = [memory_type]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY importance DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_tag(
        self, tag: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Find active memories by tag and optional namespace."""
        query = """SELECT m.* FROM memories m
               JOIN memory_tags t ON m.id = t.memory_id
               WHERE t.tag = ? AND m.is_active = 1"""
        params: list[Any] = [tag]
        if namespace is not None:
            query += " AND m.namespace = ?"
            params.append(namespace)
        query += " ORDER BY m.importance DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_created_since(
        self, since: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        """Find active memories created after a timestamp."""
        query = "SELECT * FROM memories WHERE created_at >= ? AND is_active = 1"
        params: list[Any] = [since]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embedding(
        self,
        memory_id: int,
        vector: bytes,
        model_name: str,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        if namespace is not None:
            exists = self.conn.execute(
                "SELECT 1 FROM memories WHERE id = ? AND namespace = ?",
                (memory_id, namespace),
            ).fetchone()
            if exists is None:
                return
        self.conn.execute(
            """INSERT OR REPLACE INTO embeddings (memory_id, vector, model_name)
               VALUES (?, ?, ?)""",
            (memory_id, vector, model_name),
        )
        if commit:
            self.conn.commit()

    def get_embedding(
        self, memory_id: int, *, namespace: str | None = None
    ) -> tuple[bytes, str] | None:
        """Return (vector_blob, model_name) or None."""
        query = """SELECT e.vector, e.model_name FROM embeddings e
                   JOIN memories m ON e.memory_id = m.id
                   WHERE e.memory_id = ?"""
        params: list[Any] = [memory_id]
        if namespace is not None:
            query += " AND m.namespace = ?"
            params.append(namespace)
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            return None
        return (row["vector"], row["model_name"])

    def get_all_embeddings(
        self, *, namespace: str | None = None
    ) -> list[tuple[int, bytes]]:
        """Return (memory_id, vector) for all active memories in a namespace."""
        query = """SELECT e.memory_id, e.vector
               FROM embeddings e
               JOIN memories m ON e.memory_id = m.id
               WHERE m.is_active = 1"""
        params: list[Any] = []
        if namespace is not None:
            query += " AND m.namespace = ?"
            params.append(namespace)
        rows = self.conn.execute(query, params).fetchall()
        return [(r["memory_id"], r["vector"]) for r in rows]

    def get_embedding_count(self, *, namespace: str | None = None) -> int:
        query = """SELECT COUNT(*) FROM embeddings e
                   JOIN memories m ON e.memory_id = m.id"""
        params: list[Any] = []
        if namespace is not None:
            query += " WHERE m.namespace = ?"
            params.append(namespace)
        row = self.conn.execute(query, params).fetchone()
        return row[0]
    def record_access(
        self,
        accesses: list[tuple[int, int]],
        *,
        namespace: str | None = None,
        accessed_at: str | None = None,
        commit: bool = True,
    ) -> None:
        """Persist access timestamps/counts without exposing SQL to adapters."""
        now_iso = accessed_at or datetime.now().isoformat()
        if namespace is None:
            params = [(now_iso, count, memory_id) for memory_id, count in accesses]
            self.conn.executemany(
                "UPDATE memories SET last_accessed_at = ?, access_count = ? "
                "WHERE id = ?",
                params,
            )
        else:
            params = [
                (now_iso, count, memory_id, namespace)
                for memory_id, count in accesses
            ]
            self.conn.executemany(
                "UPDATE memories SET last_accessed_at = ?, access_count = ? "
                "WHERE id = ? AND namespace = ?",
                params,
            )
        if commit:
            self.conn.commit()

    # ------------------------------------------------------------------
    def create_session(
        self,
        label: str = "",
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (label, started_at, namespace) "
            "VALUES (?, datetime('now'), ?)",
            (label or None, namespace),
        )
        if commit:
            self.conn.commit()
        return cur.lastrowid

    def end_session(
        self,
        session_id: int,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        query = "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?"
        params: list[Any] = [session_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        self.conn.execute(query, params)
        if commit:
            self.conn.commit()

    def link_memory_to_session(
        self,
        session_id: int,
        memory_id: int,
        turn_index: int | None = None,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        if namespace is not None:
            valid = self.conn.execute(
                """SELECT 1 FROM sessions s JOIN memories m
                   ON s.namespace = m.namespace
                   WHERE s.id = ? AND m.id = ? AND s.namespace = ?""",
                (session_id, memory_id, namespace),
            ).fetchone()
            if valid is None:
                return
        self.conn.execute(
            """INSERT OR IGNORE INTO session_memories
               (session_id, memory_id, turn_index) VALUES (?, ?, ?)""",
            (session_id, memory_id, turn_index),
        )
        if commit:
            self.conn.commit()

    def get_session_memories(
        self, session_id: int, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        query = """SELECT m.* FROM memories m
               JOIN session_memories sm ON m.id = sm.memory_id
               JOIN sessions s ON s.id = sm.session_id
               WHERE sm.session_id = ?"""
        params: list[Any] = [session_id]
        if namespace is not None:
            query += " AND s.namespace = ? AND m.namespace = ?"
            params.extend([namespace, namespace])
        query += " ORDER BY sm.turn_index ASC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_recent_sessions(
        self, limit: int = 10, *, namespace: str | None = None
    ) -> list[SessionRecord]:
        query = "SELECT * FROM sessions"
        params: list[Any] = []
        if namespace is not None:
            query += " WHERE namespace = ?"
            params.append(namespace)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            memory_query = """SELECT sm.memory_id FROM session_memories sm
                              JOIN memories m ON m.id = sm.memory_id
                              WHERE sm.session_id = ?"""
            memory_params: list[Any] = [row["id"]]
            if namespace is not None:
                memory_query += " AND m.namespace = ?"
                memory_params.append(namespace)
            mem_rows = self.conn.execute(memory_query, memory_params).fetchall()
            result.append(
                SessionRecord(
                    id=row["id"],
                    label=row["label"] or "",
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    memory_ids=[m["memory_id"] for m in mem_rows],
                    namespace=row["namespace"],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def search_keywords(
        self,
        query: str,
        limit: int = 20,
        *,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:
        """Naive SQL LIKE search, optionally constrained to a namespace."""
        pattern = f"%{query}%"
        sql = """SELECT * FROM memories
               WHERE is_active = 1
                 AND (content LIKE ? OR metadata LIKE ?)"""
        params: list[Any] = [pattern, pattern]
        if namespace is not None:
            sql += " AND namespace = ?"
            params.append(namespace)
        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_strengths(
        self,
        updates: list[tuple[float, int]],
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        """Batch update memory strengths, optionally constrained to a namespace."""
        now = datetime.now().isoformat()
        query = "UPDATE memories SET strength = ?, updated_at = ? WHERE id = ?"
        if namespace is not None:
            query += " AND namespace = ?"
            params = [(s, now, mid, namespace) for s, mid in updates]
        else:
            params = [(s, now, mid) for s, mid in updates]
        self.conn.executemany(query, params)
        if commit:
            self.conn.commit()

    def archive_below_threshold(
        self,
        threshold: float,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        """Archive memories below a strength threshold in an optional namespace."""
        query = "SELECT id, metadata FROM memories WHERE strength < ? AND is_active = 1"
        params: list[Any] = [threshold]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        rows = self.conn.execute(query, params).fetchall()
        now = datetime.now().isoformat()
        for row in rows:
            raw_metadata = row["metadata"]
            metadata = json.loads(raw_metadata) if raw_metadata else {}
            metadata["archival_reason"] = "decay"
            metadata["archived_at"] = now
            update_query = (
                "UPDATE memories SET is_active = 0, metadata = ?, updated_at = ? "
                "WHERE id = ?"
            )
            update_params: list[Any] = [json.dumps(metadata), now, row["id"]]
            if namespace is not None:
                update_query += " AND namespace = ?"
                update_params.append(namespace)
            self.conn.execute(update_query, update_params)
        if commit:
            self.conn.commit()
        return len(rows)

    def delete_archived_older_than(
        self, days: int = 90, *, namespace: str | None = None
    ) -> int:
        """Hard-delete archived memories older than N days in an optional namespace."""
        query = """DELETE FROM memories
               WHERE is_active = 0
                 AND updated_at < datetime('now', ?)"""
        params: list[Any] = [f"-{days} days"]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        cur = self.conn.execute(query, params)
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        # Tags are in a separate table, always fetch from there
        tags_rows = self.conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY rowid",
            (row["id"],),
        ).fetchall()
        tags = [t["tag"] for t in tags_rows]
        raw_metadata = row["metadata"]
        if raw_metadata is None:
            metadata = {}
        elif isinstance(raw_metadata, str):
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        else:
            metadata = raw_metadata
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=float(row["importance"]),
            strength=float(row["strength"]),
            access_count=int(row["access_count"]),
            last_accessed_at=row["last_accessed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
            tags=tags,
            is_active=bool(row["is_active"]),
            namespace=row["namespace"],
            confidence=(
                float(row["confidence"])
                if row["confidence"] is not None
                else None
            ),
            sensitivity=row["sensitivity"],
            source=row["source"],
            superseded_by=row["superseded_by"],
            last_decision_reason=row["last_decision_reason"],
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")
        self.conn.commit()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
