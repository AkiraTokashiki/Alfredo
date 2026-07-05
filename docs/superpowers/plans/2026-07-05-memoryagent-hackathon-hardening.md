# MemoryAgent Hackathon Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing MemoryAgent so it deduplicates repeated memories, supersedes stale preferences, honors explicit forget requests, and recalls critical memories within a bounded context window.

**Architecture:** Keep the existing SQLite store, embedding engine, forgetting curve, retrieval engine, and orchestrator. Add deterministic consolidation before memory writes and context-budget packing after retrieval. Use existing `metadata` and `is_active` fields; do not add a migration.

**Tech Stack:** Python 3.11+, SQLite stdlib, dataclasses, sentence-transformers through the existing `EmbeddingEngine`, NumPy through existing retrieval code, pytest.

---

## Version-control note

`E:/CODE/MemoryAgent` is not currently a Git repository. Do not run commit steps unless the user initializes Git first. Use targeted pytest commands as the checkpoint after each task.

## File map

- Create: `src/memory_agent/core/consolidation.py`
  - Defines `ConsolidationAction`, `ConsolidationDecision`, `MemoryConsolidator`.
  - Performs duplicate detection, preference supersession, and explicit forget matching.
- Create: `src/memory_agent/core/context_budget.py`
  - Defines `RecallPacket` and `ContextBudgetPacker`.
  - Packs `SearchResult` objects into a character budget with selected/omitted accounting.
- Modify: `src/memory_agent/core/config.py`
  - Add consolidation thresholds and context-budget settings.
- Modify: `src/memory_agent/core/memory_store.py`
  - Add `archive_memory()` with archival metadata.
  - Update `archive_below_threshold()` to stamp `archival_reason=decay`.
- Modify: `src/memory_agent/core/retrieval.py`
  - Add `candidate_k` support while preserving current `top_k` behavior.
- Modify: `src/memory_agent/agent/decision.py`
  - Add explicit forget phrase detection.
  - Add lightweight topic/polarity metadata for preferences.
- Modify: `src/memory_agent/agent/orchestrator.py`
  - Instantiate consolidator and budget packer.
  - Route extracted memories through consolidation.
  - Return `recall_packet` and use selected memories for formatted context.
- Create: `tests/test_consolidation.py`
- Create: `tests/test_context_budget.py`
- Modify: `tests/test_agent.py`
- Create: `examples/demo_hackathon.py`
- Modify: `README.md`

---

### Task 1: Add configuration and models for hardening

**Files:**
- Modify: `src/memory_agent/core/config.py`
- Modify: `src/memory_agent/models.py`
- Test: `tests/test_context_budget.py`

- [ ] **Step 1: Write model/config smoke tests**

Create `tests/test_context_budget.py` with this initial content:

```python
"""Tests for context-budget recall packing."""

from __future__ import annotations

from memory_agent.core.config import ConsolidationConfig, RetrievalConfig
from memory_agent.models import MemoryRecord, SearchResult


def test_retrieval_config_has_context_budget_defaults():
    config = RetrievalConfig()
    assert config.candidate_k >= config.top_k
    assert config.context_budget_chars > 0
    assert config.reserved_context_chars >= 0


def test_consolidation_config_has_thresholds():
    config = ConsolidationConfig()
    assert 0.0 < config.duplicate_similarity_threshold <= 1.0
    assert 0.0 < config.supersede_similarity_threshold <= 1.0
    assert config.explicit_forget_min_score >= 0.0


def test_search_result_estimated_chars_defaults_to_content_length():
    memory = MemoryRecord(content="El usuario prefiere: Rust", memory_type="preference")
    result = SearchResult(memory=memory, score=0.9)
    assert result.estimated_chars == len(memory.content)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_context_budget.py -v
```

Expected: fails because `candidate_k`, `context_budget_chars`, `reserved_context_chars`, consolidation thresholds, and `SearchResult.estimated_chars` do not exist yet.

- [ ] **Step 3: Update config and SearchResult**

Modify `src/memory_agent/core/config.py`:

```python
@dataclass
class RetrievalConfig:
    """Retrieval scoring weights."""

    # Weight for semantic similarity (cosine distance)
    w_semantic: float = 0.40

    # Weight for recency (1 / (1 + hours))
    w_recency: float = 0.20

    # Weight for importance
    w_importance: float = 0.20

    # Weight for recall strength
    w_strength: float = 0.20

    # Maximum memories to return to callers using classic retrieval
    top_k: int = 10

    # Candidate pool size before context-budget packing
    candidate_k: int = 20

    # Character budget for formatted recollections
    context_budget_chars: int = 2400

    # Reserved characters for the current prompt and instructions
    reserved_context_chars: int = 600

    # MMR diversity lambda (0 = pure relevance, 1 = pure diversity)
    mmr_lambda: float = 0.5

    # Minimum score to include (filters noise)
    min_score: float = 0.05
```

Modify `src/memory_agent/core/config.py` consolidation config:

```python
@dataclass
class ConsolidationConfig:
    """Memory consolidation parameters."""

    # After this many turns, consolidate short-term → long-term
    consolidation_interval: int = 5

    # Minimum importance to auto-consolidate
    auto_consolidate_threshold: float = 0.3

    # How often (in turns) to run full forgetting decay
    decay_interval: int = 3

    # Max memories to keep in short-term working context
    working_context_size: int = 15

    # Semantic threshold for treating a candidate as duplicate
    duplicate_similarity_threshold: float = 0.88

    # Semantic threshold for detecting same-topic preference replacement
    supersede_similarity_threshold: float = 0.55

    # Minimum score for explicit forget matches
    explicit_forget_min_score: float = 0.35
```

Modify `src/memory_agent/models.py` `SearchResult`:

```python
@dataclass
class SearchResult:
    """A memory returned from a search query."""

    memory: MemoryRecord
    score: float
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    strength_score: float = 0.0

    @property
    def estimated_chars(self) -> int:
        """Approximate formatted context cost for this memory."""
        return len(self.memory.content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_context_budget.py -v
```

Expected: 3 tests pass.

---

### Task 2: Add context-budget packing

**Files:**
- Create: `src/memory_agent/core/context_budget.py`
- Modify: `tests/test_context_budget.py`

- [ ] **Step 1: Extend context-budget tests**

Append these tests to `tests/test_context_budget.py`:

```python
from memory_agent.core.context_budget import ContextBudgetPacker


def _result(content: str, score: float, importance: float = 0.5) -> SearchResult:
    memory = MemoryRecord(content=content, memory_type="preference", importance=importance)
    return SearchResult(memory=memory, score=score, importance_score=importance)


def test_context_budget_selects_memories_that_fit():
    packer = ContextBudgetPacker(budget_chars=80, reserved_chars=10)
    results = [
        _result("critical short preference", 0.95, 0.9),
        _result("x" * 200, 0.99, 1.0),
        _result("minor memory", 0.2, 0.2),
    ]

    packet = packer.pack(results)

    assert [r.memory.content for r in packet.selected] == ["critical short preference", "minor memory"]
    assert packet.omitted[0].memory.content == "x" * 200
    assert packet.used_chars <= packet.available_chars
    assert "selected" in packet.reasons[id(packet.selected[0])]
    assert "too large" in packet.reasons[id(packet.omitted[0])]


def test_context_budget_returns_empty_when_reserved_exhausts_budget():
    packer = ContextBudgetPacker(budget_chars=10, reserved_chars=10)
    packet = packer.pack([_result("important", 1.0, 1.0)])
    assert packet.selected == []
    assert len(packet.omitted) == 1
    assert packet.available_chars == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_context_budget.py -v
```

Expected: import failure for `memory_agent.core.context_budget`.

- [ ] **Step 3: Create context budget module**

Create `src/memory_agent/core/context_budget.py`:

```python
"""Context-budget-aware recall packing."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory_agent.models import SearchResult


@dataclass
class RecallPacket:
    """Selected and omitted recollections for a bounded context window."""

    selected: list[SearchResult]
    omitted: list[SearchResult]
    budget_chars: int
    reserved_chars: int
    used_chars: int
    reasons: dict[int, str] = field(default_factory=dict)

    @property
    def available_chars(self) -> int:
        return max(0, self.budget_chars - self.reserved_chars)


class ContextBudgetPacker:
    """Packs ranked memories into a character budget."""

    def __init__(self, budget_chars: int, reserved_chars: int = 0) -> None:
        self.budget_chars = max(0, budget_chars)
        self.reserved_chars = max(0, reserved_chars)

    def pack(self, results: list[SearchResult]) -> RecallPacket:
        available = max(0, self.budget_chars - self.reserved_chars)
        selected: list[SearchResult] = []
        omitted: list[SearchResult] = []
        reasons: dict[int, str] = {}
        used = 0

        ranked = sorted(
            results,
            key=lambda r: (
                r.score / max(r.estimated_chars, 1),
                r.score,
                r.memory.importance,
            ),
            reverse=True,
        )

        for result in ranked:
            cost = result.estimated_chars
            result_key = id(result)
            if cost > available:
                omitted.append(result)
                reasons[result_key] = (
                    f"omitted: too large for available budget "
                    f"({cost} chars > {available} chars)"
                )
                continue
            if used + cost <= available:
                selected.append(result)
                used += cost
                reasons[result_key] = (
                    f"selected: score={result.score:.3f}, "
                    f"cost={cost}, used={used}/{available}"
                )
            else:
                omitted.append(result)
                reasons[result_key] = (
                    f"omitted: remaining budget too small "
                    f"({available - used} chars left, needs {cost})"
                )

        return RecallPacket(
            selected=selected,
            omitted=omitted,
            budget_chars=self.budget_chars,
            reserved_chars=self.reserved_chars,
            used_chars=used,
            reasons=reasons,
        )
```

- [ ] **Step 4: Run context-budget tests**

Run:

```bash
python -m pytest tests/test_context_budget.py -v
```

Expected: all context-budget tests pass.

---

### Task 3: Add archival metadata support

**Files:**
- Modify: `src/memory_agent/core/memory_store.py`
- Create: `tests/test_consolidation.py`

- [ ] **Step 1: Write archival metadata tests**

Create `tests/test_consolidation.py` with this initial content:

```python
"""Tests for memory consolidation and stale-memory archival."""

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


def test_archive_memory_records_reason(store: MemoryStore):
    memory_id = store.add_memory(MemoryRecord(content="old preference", metadata={"topic": "language"}))

    store.archive_memory(memory_id, reason="superseded", metadata={"superseded_by": 99})

    archived = store.get_memory(memory_id)
    assert archived is not None
    assert archived.is_active is False
    assert archived.metadata["archival_reason"] == "superseded"
    assert archived.metadata["superseded_by"] == 99


def test_archive_below_threshold_records_decay_reason(store: MemoryStore):
    memory_id = store.add_memory(MemoryRecord(content="weak", strength=0.01))

    count = store.archive_below_threshold(0.05)

    archived = store.get_memory(memory_id)
    assert count == 1
    assert archived is not None
    assert archived.is_active is False
    assert archived.metadata["archival_reason"] == "decay"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_consolidation.py -v
```

Expected: fails because `archive_memory()` does not exist and threshold archival does not stamp metadata.

- [ ] **Step 3: Add archival helpers to MemoryStore**

Modify `src/memory_agent/core/memory_store.py`.

Add this method after `delete_memory()`:

```python
    def archive_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> None:
        """Archive one memory and preserve the reason in metadata."""
        memory = self.get_memory(memory_id)
        if memory is None:
            return

        merged_metadata = dict(memory.metadata)
        merged_metadata["archival_reason"] = reason
        if metadata:
            merged_metadata.update(metadata)

        memory.is_active = False
        memory.metadata = merged_metadata
        self.update_memory(memory, commit=commit)
```

Replace `archive_below_threshold()` with:

```python
    def archive_below_threshold(self, threshold: float, *, commit: bool = True) -> int:
        """Archive memories whose strength has fallen below threshold.

        Returns the number of memories archived.
        """
        rows = self.conn.execute(
            "SELECT id, metadata FROM memories WHERE strength < ? AND is_active = 1",
            (threshold,),
        ).fetchall()
        now = datetime.now().isoformat()

        for row in rows:
            raw_metadata = row["metadata"]
            if raw_metadata:
                metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            else:
                metadata = {}
            metadata["archival_reason"] = "decay"
            metadata["archived_at"] = now
            self.conn.execute(
                """UPDATE memories
                   SET is_active = 0, metadata = ?, updated_at = ?
                   WHERE id = ?""",
                (json.dumps(metadata), now, row["id"]),
            )

        if commit:
            self.conn.commit()
        return len(rows)
```

- [ ] **Step 4: Run archival tests**

Run:

```bash
python -m pytest tests/test_consolidation.py -v
```

Expected: 2 tests pass.

---

### Task 4: Add deterministic consolidation

**Files:**
- Create: `src/memory_agent/core/consolidation.py`
- Modify: `tests/test_consolidation.py`

- [ ] **Step 1: Add consolidation behavior tests**

Append these tests to `tests/test_consolidation.py`:

```python
from memory_agent.core.config import ConsolidationConfig
from memory_agent.core.consolidation import ConsolidationAction, MemoryConsolidator


class FakeSimilarity:
    def __init__(self, scores: dict[tuple[str, str], float] | None = None) -> None:
        self.scores = scores or {}

    def similarity(self, left: str, right: str) -> float:
        return self.scores.get((left, right), self.scores.get((right, left), 0.0))


def test_duplicate_preference_reinforces_existing(store: MemoryStore):
    existing = MemoryRecord(content="El usuario prefiere: Python", memory_type="preference", strength=0.4)
    existing_id = store.add_memory(existing)
    candidate = MemoryRecord(content="El usuario prefiere: programar en Python", memory_type="preference")
    similarity = FakeSimilarity({(existing.content, candidate.content): 0.93})
    consolidator = MemoryConsolidator(store, similarity, ConsolidationConfig())

    decision = consolidator.consolidate(candidate)

    refreshed = store.get_memory(existing_id)
    assert decision.action is ConsolidationAction.REINFORCE
    assert decision.existing_memory_id == existing_id
    assert refreshed is not None
    assert refreshed.strength > 0.4
    assert store.count_memories() == 1


def test_conflicting_preference_supersedes_existing(store: MemoryStore):
    old = MemoryRecord(
        content="El usuario prefiere: Python",
        memory_type="preference",
        metadata={"topic": "python", "polarity": "positive"},
    )
    old_id = store.add_memory(old)
    new = MemoryRecord(
        content="Al usuario no le gusta: Python",
        memory_type="preference",
        metadata={"topic": "python", "polarity": "negative"},
    )
    similarity = FakeSimilarity({(old.content, new.content): 0.70})
    consolidator = MemoryConsolidator(store, similarity, ConsolidationConfig())

    decision = consolidator.consolidate(new)

    archived_old = store.get_memory(old_id)
    stored_new = store.get_memory(decision.new_memory_id or -1)
    assert decision.action is ConsolidationAction.UPDATE
    assert archived_old is not None
    assert archived_old.is_active is False
    assert archived_old.metadata["archival_reason"] == "superseded"
    assert stored_new is not None
    assert stored_new.is_active is True
    assert stored_new.metadata["supersedes"] == old_id


def test_explicit_forget_archives_matching_memory(store: MemoryStore):
    memory_id = store.add_memory(MemoryRecord(content="El usuario prefiere: Python", memory_type="preference"))
    similarity = FakeSimilarity({("forget python", "El usuario prefiere: Python"): 0.8})
    consolidator = MemoryConsolidator(store, similarity, ConsolidationConfig())

    archived_count = consolidator.forget_matching("forget python")

    archived = store.get_memory(memory_id)
    assert archived_count == 1
    assert archived is not None
    assert archived.is_active is False
    assert archived.metadata["archival_reason"] == "explicit_user_request"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_consolidation.py -v
```

Expected: import failure for `memory_agent.core.consolidation`.

- [ ] **Step 3: Create consolidation module**

Create `src/memory_agent/core/consolidation.py`:

```python
"""Deterministic memory consolidation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from memory_agent.core.config import ConsolidationConfig
from memory_agent.core.memory_store import MemoryStore
from memory_agent.models import MemoryRecord


class TextSimilarity(Protocol):
    def similarity(self, left: str, right: str) -> float:
        """Return similarity in the inclusive range [0.0, 1.0]."""


class ConsolidationAction(str, Enum):
    CREATE = "create"
    REINFORCE = "reinforce"
    UPDATE = "update"
    IGNORE = "ignore"


@dataclass
class ConsolidationDecision:
    action: ConsolidationAction
    candidate: MemoryRecord
    existing_memory_id: int | None = None
    new_memory_id: int | None = None
    reason: str = ""


class MemoryConsolidator:
    """Consolidates extracted memories before they are stored."""

    def __init__(
        self,
        store: MemoryStore,
        similarity: TextSimilarity,
        config: ConsolidationConfig,
    ) -> None:
        self.store = store
        self.similarity = similarity
        self.config = config

    def consolidate(self, candidate: MemoryRecord) -> ConsolidationDecision:
        if candidate.importance < self.config.auto_consolidate_threshold:
            return ConsolidationDecision(
                action=ConsolidationAction.IGNORE,
                candidate=candidate,
                reason="candidate below auto-consolidation threshold",
            )

        active = self.store.get_all_active_memories()
        same_type = [m for m in active if m.memory_type == candidate.memory_type]
        best = self._best_match(candidate, same_type)

        if best is None:
            new_id = self.store.add_memory(candidate, commit=False)
            return ConsolidationDecision(
                action=ConsolidationAction.CREATE,
                candidate=candidate,
                new_memory_id=new_id,
                reason="no active memory matched candidate",
            )

        existing, score = best
        if score >= self.config.duplicate_similarity_threshold:
            existing.strength = min(1.0, existing.strength + 0.15)
            existing.access_count += 1
            existing.last_accessed_at = datetime.now().isoformat()
            existing.metadata = {
                **existing.metadata,
                "consolidation_action": "reinforce",
                "last_duplicate": candidate.content,
            }
            self.store.update_memory(existing, commit=False)
            return ConsolidationDecision(
                action=ConsolidationAction.REINFORCE,
                candidate=candidate,
                existing_memory_id=existing.id,
                reason=f"duplicate similarity {score:.3f}",
            )

        if self._supersedes(existing, candidate, score):
            new_id = self.store.add_memory(candidate, commit=False)
            candidate.metadata = {
                **candidate.metadata,
                "supersedes": existing.id,
                "consolidation_action": "update",
            }
            self.store.update_memory(candidate, commit=False)
            self.store.archive_memory(
                existing.id or -1,
                reason="superseded",
                metadata={
                    "superseded_by": new_id,
                    "superseded_at": datetime.now().isoformat(),
                },
                commit=False,
            )
            return ConsolidationDecision(
                action=ConsolidationAction.UPDATE,
                candidate=candidate,
                existing_memory_id=existing.id,
                new_memory_id=new_id,
                reason=f"candidate superseded active memory at similarity {score:.3f}",
            )

        new_id = self.store.add_memory(candidate, commit=False)
        return ConsolidationDecision(
            action=ConsolidationAction.CREATE,
            candidate=candidate,
            new_memory_id=new_id,
            reason=f"best similarity {score:.3f} did not trigger consolidation",
        )

    def forget_matching(self, query: str) -> int:
        archived = 0
        for memory in self.store.get_all_active_memories():
            score = self.similarity.similarity(query, memory.content)
            if score >= self.config.explicit_forget_min_score and memory.id is not None:
                self.store.archive_memory(
                    memory.id,
                    reason="explicit_user_request",
                    metadata={"forget_query": query},
                    commit=False,
                )
                archived += 1
        return archived

    def _best_match(
        self,
        candidate: MemoryRecord,
        memories: list[MemoryRecord],
    ) -> tuple[MemoryRecord, float] | None:
        best_memory: MemoryRecord | None = None
        best_score = 0.0
        for memory in memories:
            score = self.similarity.similarity(memory.content, candidate.content)
            if score > best_score:
                best_memory = memory
                best_score = score
        if best_memory is None:
            return None
        return best_memory, best_score

    def _supersedes(self, existing: MemoryRecord, candidate: MemoryRecord, score: float) -> bool:
        if existing.memory_type != "preference" or candidate.memory_type != "preference":
            return False
        if score < self.config.supersede_similarity_threshold:
            return False
        existing_polarity = existing.metadata.get("polarity")
        candidate_polarity = candidate.metadata.get("polarity")
        if existing_polarity and candidate_polarity and existing_polarity != candidate_polarity:
            return True
        existing_topic = existing.metadata.get("topic")
        candidate_topic = candidate.metadata.get("topic")
        return bool(existing_topic and candidate_topic and existing_topic == candidate_topic)
```

- [ ] **Step 4: Run consolidation tests**

Run:

```bash
python -m pytest tests/test_consolidation.py -v
```

Expected: archival and consolidation tests pass.

---

### Task 5: Add extraction metadata and explicit forget detection

**Files:**
- Modify: `src/memory_agent/agent/decision.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Add decision tests**

Append these tests to `tests/test_agent.py`:

```python
    def test_preference_extraction_adds_topic_and_polarity(self):
        from memory_agent.agent.decision import extract_from_input

        memories = extract_from_input("Me gusta Python")

        assert memories
        assert memories[0].metadata["topic"] == "python"
        assert memories[0].metadata["polarity"] == "positive"

    def test_negative_preference_extraction_adds_negative_polarity(self):
        from memory_agent.agent.decision import extract_from_input

        memories = extract_from_input("No me gusta Python")

        assert memories
        assert memories[0].metadata["topic"] == "python"
        assert memories[0].metadata["polarity"] == "negative"

    def test_explicit_forget_detection(self):
        from memory_agent.agent.decision import extract_forget_query

        assert extract_forget_query("forget python") == "python"
        assert extract_forget_query("olvida que me gusta python") == "me gusta python"
        assert extract_forget_query("Me gusta Python") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_agent.py::TestMemoryAgent::test_preference_extraction_adds_topic_and_polarity tests/test_agent.py::TestMemoryAgent::test_negative_preference_extraction_adds_negative_polarity tests/test_agent.py::TestMemoryAgent::test_explicit_forget_detection -v
```

Expected: failures because metadata and `extract_forget_query()` are absent.

- [ ] **Step 3: Update decision extraction**

Add helpers near the top of `src/memory_agent/agent/decision.py` after the pattern constants:

```python
_FORGET_PATTERNS: list[str] = [
    r"(?:forget|remove|delete)\s+(?:that\s+)?(.+)",
    r"(?:olvida|borra|elimina)\s+(?:que\s+)?(.+)",
]


def _topic_from_content(content: str) -> str:
    topic = content.lower().strip()
    topic = re.sub(r"^(el usuario prefiere:|al usuario no le gusta:|hecho:)\s*", "", topic)
    topic = re.sub(r"[^a-z0-9áéíóúñü\s]+", "", topic)
    return " ".join(topic.split())


def extract_forget_query(text: str) -> str | None:
    """Extract the target of an explicit forget request."""
    text_lower = text.lower().strip()
    for pattern in _FORGET_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            query = match.group(1).strip().rstrip(".,!?")
            return query or None
    return None
```

Update `extract_from_input()` preference append block:

```python
    for content, mem_type, importance in extract_preferences(text):
        polarity = "negative" if "no le gusta" in content.lower() else "positive"
        memories.append(MemoryRecord(
            content=content,
            memory_type=mem_type,
            importance=importance,
            metadata={
                "topic": _topic_from_content(content),
                "polarity": polarity,
            },
            tags=["extracted", "preference"],
        ))
```

Update fact append block:

```python
    for content, mem_type, importance in extract_facts(text):
        memories.append(MemoryRecord(
            content=content,
            memory_type=mem_type,
            importance=importance,
            metadata={"topic": _topic_from_content(content)},
            tags=["extracted", "fact"],
        ))
```

- [ ] **Step 4: Run targeted decision tests**

Run:

```bash
python -m pytest tests/test_agent.py::TestMemoryAgent::test_preference_extraction_adds_topic_and_polarity tests/test_agent.py::TestMemoryAgent::test_negative_preference_extraction_adds_negative_polarity tests/test_agent.py::TestMemoryAgent::test_explicit_forget_detection -v
```

Expected: 3 tests pass.

---

### Task 6: Integrate consolidation and context budget into the orchestrator

**Files:**
- Modify: `src/memory_agent/core/retrieval.py`
- Modify: `src/memory_agent/agent/orchestrator.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Add integration tests**

Append these tests to `tests/test_agent.py`:

```python
    def test_duplicate_preference_is_not_stored_twice(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta Python")
        first_total = agent.state.total_memories

        result = agent.perceive("Me gusta programar en Python")

        active_preferences = [
            m for m in agent.store.get_all_active_memories()
            if m.memory_type == "preference" and "python" in m.content.lower()
        ]
        assert len(active_preferences) == 1
        assert result["consolidation_decisions"]
        assert agent.state.total_memories <= first_total + 1

    def test_changed_preference_supersedes_old_preference(self, agent: MemoryAgent):
        agent.init_session("session 1")
        agent.perceive("Me gusta Python")
        agent.end_session()

        agent.init_session("session 2")
        agent.perceive("No me gusta Python")

        all_memories = [agent.store.get_memory(m.id) for m in agent.store.get_all_active_memories() if m.id]
        active_python_preferences = [
            m for m in all_memories
            if m is not None and m.memory_type == "preference" and "python" in m.content.lower()
        ]
        assert len(active_python_preferences) == 1
        assert "no le gusta" in active_python_preferences[0].content.lower()

        archived_count = agent.store.count_memories(active_only=False) - agent.store.count_memories(active_only=True)
        assert archived_count >= 1

    def test_explicit_forget_removes_memory_from_default_recall(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta Python")

        result = agent.perceive("forget python")

        assert result["archived"] >= 1
        active_contents = [m.content.lower() for m in agent.store.get_all_active_memories()]
        assert not any("prefiere: python" in content for content in active_contents)

    def test_perceive_returns_recall_packet(self, agent: MemoryAgent):
        agent.init_session()
        agent.perceive("Me gusta Python")

        result = agent.perceive("Que lenguaje me gusta?")

        assert "recall_packet" in result
        assert result["recall_packet"].used_chars <= result["recall_packet"].available_chars
```

- [ ] **Step 2: Run integration tests to verify they fail**

Run:

```bash
python -m pytest tests/test_agent.py::TestMemoryAgent::test_duplicate_preference_is_not_stored_twice tests/test_agent.py::TestMemoryAgent::test_changed_preference_supersedes_old_preference tests/test_agent.py::TestMemoryAgent::test_explicit_forget_removes_memory_from_default_recall tests/test_agent.py::TestMemoryAgent::test_perceive_returns_recall_packet -v
```

Expected: failures because orchestrator does not call consolidation, explicit forget, or context budget packer.

- [ ] **Step 3: Add candidate pool support to retrieval**

In `src/memory_agent/core/retrieval.py`, change the `retrieve()` signature to:

```python
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        memory_type: str | None = None,
        min_score: float | None = None,
        use_mmr: bool = True,
        mmr_lambda: float | None = None,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
```

After `top_k = top_k or self.config.top_k`, add:

```python
        candidate_k = candidate_k or top_k
        candidate_k = max(candidate_k, top_k)
```

Replace the MMR slice block with:

```python
        # Apply MMR diversity if requested and we have more results than candidate_k
        if use_mmr and len(results) > candidate_k:
            results = self._mmr_diversify(results, query_vec, mmr_lambda, candidate_k)
        else:
            results = results[:candidate_k]

        results = results[:top_k] if candidate_k == top_k else results
```

This preserves existing callers because `candidate_k` defaults to `top_k`.

- [ ] **Step 4: Add embedding similarity adapter and orchestrator integration**

In `src/memory_agent/agent/orchestrator.py`, add imports:

```python
from memory_agent.agent.decision import extract_forget_query
from memory_agent.core.consolidation import ConsolidationDecision, MemoryConsolidator
from memory_agent.core.context_budget import ContextBudgetPacker, RecallPacket
```

Add this helper class above `MemoryAgent`:

```python
class EmbeddingSimilarity:
    """Text similarity adapter backed by the configured embedding engine."""

    def __init__(self, embeddings: EmbeddingEngine) -> None:
        self.embeddings = embeddings

    def similarity(self, left: str, right: str) -> float:
        left_vec = self.embeddings.decode_vector(self.embeddings.encode(left))
        right_vec = self.embeddings.decode_vector(self.embeddings.encode(right))
        return self.embeddings.cosine_similarity(left_vec, right_vec)
```

In `MemoryAgent.__init__()`, after retrieval initialization, add:

```python
        self.consolidator = MemoryConsolidator(
            self.store,
            EmbeddingSimilarity(self.embeddings),
            self.config.consolidation,
        )
        self.context_packer = ContextBudgetPacker(
            budget_chars=self.config.retrieval.context_budget_chars,
            reserved_chars=self.config.retrieval.reserved_context_chars,
        )
```

In `perceive()`, initialize decision state after `archived = 0`:

```python
        consolidation_decisions: list[ConsolidationDecision] = []
        recall_packet: RecallPacket | None = None
```

Replace the extraction storage loop with:

```python
        forget_query = extract_forget_query(user_input)
        if forget_query:
            archived += self.consolidator.forget_matching(forget_query)

        extracted = extract_from_input(user_input)
        for mem in extracted:
            decision = self.consolidator.consolidate(mem)
            consolidation_decisions.append(decision)
            if decision.new_memory_id is not None:
                new_memories.append(mem)
```

Replace the retrieval call with:

```python
            candidate_recollections = self.retrieval.retrieve(
                query=user_input,
                top_k=self.config.retrieval.top_k,
                candidate_k=self.config.retrieval.candidate_k,
                use_mmr=True,
            )
            recall_packet = self.context_packer.pack(candidate_recollections)
            recollections = recall_packet.selected
```

Add to the returned dict:

```python
            "recall_packet": recall_packet,
            "consolidation_decisions": consolidation_decisions,
```

- [ ] **Step 5: Run integration tests**

Run:

```bash
python -m pytest tests/test_agent.py::TestMemoryAgent::test_duplicate_preference_is_not_stored_twice tests/test_agent.py::TestMemoryAgent::test_changed_preference_supersedes_old_preference tests/test_agent.py::TestMemoryAgent::test_explicit_forget_removes_memory_from_default_recall tests/test_agent.py::TestMemoryAgent::test_perceive_returns_recall_packet -v
```

Expected: 4 tests pass.

---

### Task 7: Add judge-facing demo

**Files:**
- Create: `examples/demo_hackathon.py`
- Modify: `README.md`

- [ ] **Step 1: Create demo script**

Create `examples/demo_hackathon.py`:

```python
"""Hackathon demo: persistent memory, stale preference replacement, and context budget."""

from __future__ import annotations

from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.models import MemoryRecord


DB_PATH = Path("hackathon_demo.db")


def print_turn(title: str, result: dict) -> None:
    print(f"\n=== {title} ===")
    print(result["recollection_text"] or "[no recollections]")
    print(f"active memories: {result['total_memories']}")
    print(f"archived this turn: {result['archived']}")
    packet = result.get("recall_packet")
    if packet is not None:
        print(f"context budget: {packet.used_chars}/{packet.available_chars} chars")
        print(f"omitted memories: {len(packet.omitted)}")


def main() -> None:
    DB_PATH.unlink(missing_ok=True)
    agent = MemoryAgent(db_path=DB_PATH)

    agent.init_session("session 1")
    print_turn("Session 1: learn preferences", agent.perceive("Me gusta Python y prefiero respuestas concisas"))
    agent.end_session()

    agent.init_session("session 2")
    print_turn("Session 2: recall preference", agent.perceive("Que lenguaje me gusta?"))
    agent.end_session()

    agent.init_session("session 3")
    print_turn("Session 3: update stale preference", agent.perceive("No me gusta Python"))
    agent.end_session()

    agent.init_session("session 4")
    for idx in range(20):
        agent.store_memory(MemoryRecord(content=f"ruido de baja importancia {idx}", importance=0.1))
    print_turn("Session 4: bounded recall after noise", agent.perceive("Que recuerdas de mis preferencias?"))

    stats = agent.get_stats()
    print("\n=== Stats ===")
    print(f"active: {stats['total_active']}")
    print(f"archived: {stats['archived']}")
    print(f"types: {stats['type_distribution']}")

    agent.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run demo**

Run:

```bash
python examples/demo_hackathon.py
```

Expected output contains these labels:

```text
Session 1: learn preferences
Session 2: recall preference
Session 3: update stale preference
Session 4: bounded recall after noise
Stats
```

- [ ] **Step 3: Update README demo section**

In `README.md`, add this under the demo section:

```markdown
### Hackathon demo

```bash
python examples/demo_hackathon.py
```

This demo shows the complete memory lifecycle:

1. learns a user preference in one session;
2. recalls it in a later session;
3. archives the stale preference when the user changes it;
4. keeps critical memories inside a bounded recall context even after low-importance noise is added;
5. prints active/archived memory stats for judging.
```

- [ ] **Step 4: Run demo again**

Run:

```bash
python examples/demo_hackathon.py
```

Expected: demo completes without traceback and shows an archived count of at least 1 after the preference update.

---

### Task 8: Final targeted verification

**Files:**
- No code changes unless a verification failure exposes a bug.

- [ ] **Step 1: Run new focused tests**

Run:

```bash
python -m pytest tests/test_consolidation.py tests/test_context_budget.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run affected orchestrator tests**

Run:

```bash
python -m pytest tests/test_agent.py -v
```

Expected: all `test_agent.py` tests pass.

- [ ] **Step 3: Run retrieval regression tests**

Run:

```bash
python -m pytest tests/test_retrieval.py -v
```

Expected: all retrieval tests pass, proving `candidate_k` preserved classic retrieval behavior.

- [ ] **Step 4: Run store regression tests**

Run:

```bash
python -m pytest tests/test_memory_store.py tests/test_forgetting.py -v
```

Expected: all storage and forgetting tests pass, including decay archival behavior.

- [ ] **Step 5: Run hackathon demo**

Run:

```bash
python examples/demo_hackathon.py
```

Expected: output demonstrates cross-session recall, preference update, archived stale memory, context-budget usage, and stats.

---

## Self-review coverage

Spec requirement coverage:

- Efficient storage/retrieval: Task 6 preserves current retrieval and adds candidate pool support.
- Timely forgetting: Tasks 3, 4, 5, and 6 add archival reasons, supersession, and explicit forget.
- Critical recall in limited context: Tasks 1, 2, and 6 add `RecallPacket` and context-budget packing.
- Cross-session improving decisions: Task 6 integration tests and Task 7 demo cover preference update across sessions.
- Hackathon presentation: Task 7 adds a judge-facing demo and README section.

No database migration is included because existing `metadata` and `is_active` support the required behavior.
