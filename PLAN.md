# MemoryAgent — Implementation Plan

> **Goal:** Build an agent with persistent memory that accumulates experience autonomously, remembers user preferences, and makes increasingly accurate decisions across multi-turn and cross-session interactions — with proper forgetting of obsolete info and efficient retrieval within limited context windows.

**Architecture:** Three-layer memory system (Short-term → Consolidation → Long-term) with Ebbinghaus forgetting curve, semantic embeddings for retrieval, and importance-weighted retention. SQLite backend for persistence, sentence-transformers for embeddings.

**Tech Stack:** Python 3.11+, SQLite (via sqlite3 stdlib), sentence-transformers, numpy, Click (CLI), pytest

---

## Project Structure

```
E:\CODE\MemoryAgent\
├── src\
│   ├── memory_agent\
│   │   ├── __init__.py
│   │   ├── core\
│   │   │   ├── __init__.py
│   │   │   ├── memory_store.py      # SQLite schemas, CRUD
│   │   │   ├── embeddings.py        # sentence-transformers wrapper
│   │   │   ├── forgetting.py        # Ebbinghaus curve, importance decay
│   │   │   ├── retrieval.py         # Semantic + recency + importance scoring
│   │   │   └── config.py            # Configuration dataclass
│   │   ├── agent\
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py      # Main agent loop
│   │   │   └── decision.py          # Context-aware decision making
│   │   ├── cli\
│   │   │   ├── __init__.py
│   │   │   └── commands.py          # Click commands
│   │   └── models.py                # Pydantic/dataclass models
│   └── memory_agent\__main__.py     # Entry point
├── tests\
│   ├── test_memory_store.py
│   ├── test_embeddings.py
│   ├── test_forgetting.py
│   ├── test_retrieval.py
│   └── test_agent.py
├── examples\
│   ├── demo_basic.py
│   └── demo_multi_session.py
├── PLAN.md
├── HISTORY.md
├── requirements.txt
├── setup.py / pyproject.toml
└── README.md
```

## Memory Schema (SQLite)

```sql
-- Core memory table
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'episodic',  -- episodic | semantic | procedural | preference
    importance REAL NOT NULL DEFAULT 0.5,          -- [0, 1] user-assigned importance
    strength REAL NOT NULL DEFAULT 1.0,            -- [0, 1] recall strength (decays via forgetting curve)
    access_count INTEGER NOT NULL DEFAULT 0,       -- how many times retrieved
    last_accessed_at TEXT,                          -- ISO timestamp
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}',                     -- JSON blob for extra attributes
    is_active INTEGER NOT NULL DEFAULT 1            -- soft delete / archival
);

-- Embeddings table (separate for efficient queries)
CREATE TABLE embeddings (
    memory_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,                           -- numpy array as pickle
    model_name TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

-- Tags for categorization
CREATE TABLE memory_tags (
    memory_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

-- Sessions (for episodic grouping)
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT
);

-- Session-memory bridge
CREATE TABLE session_memories (
    session_id INTEGER NOT NULL,
    memory_id INTEGER NOT NULL,
    turn_index INTEGER,
    PRIMARY KEY (session_id, memory_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_last_accessed ON memories(last_accessed_at);
CREATE INDEX idx_memories_active ON memories(is_active) WHERE is_active = 1;
```

## Forgetting Curve (Ebbinghaus)

strength(t) = initial_strength × e^(-t / decay_constant)

- On each retrieval: strength = min(1.0, strength + reinforcement_boost)
- On decay tick: strength ×= e^(-delta_hours / decay_hours)
- decay_hours depends on importance:
  - importance >= 0.8: decay_hours = 720 (30 days)
  - importance >= 0.5: decay_hours = 168 (7 days)
  - importance < 0.5: decay_hours = 24 (1 day)
- When strength < threshold (0.05): memory is archived (is_active = 0)

## Retrieval Scoring

total_score = w1 * semantic_similarity + w2 * recency_score + w3 * importance + w4 * strength

Where:
- semantic_similarity: cosine similarity between query embedding and memory embedding
- recency_score: 1 / (1 + hours_since_last_access) normalized
- importance: direct value from memory record
- strength: current recall strength from forgetting curve
- w1=0.4, w2=0.2, w3=0.2, w4=0.2 (configurable)

Context window optimization: top-k results with diversity penalty (MMR - Maximum Marginal Relevance) to avoid near-duplicate retrieval.

## Agent Loop (Orchestrator)

1. **Perceive** — receive input (user message)
2. **Retrieve** — search memories (semantic + importance + recency)
3. **Augment** — inject top-N memories into context as "recollection"
4. **Decide** — use recollections to inform response/action
5. **Memorize** — extract new facts, preferences, and consolidate
6. **Decay** — apply forgetting curve to all memories
7. **Repeat**

---

## Tasks

### Task 1: Project scaffolding

**Objective:** Create the project directory, venv, requirements, pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `MANIFEST.in`

### Task 2: Data models

**Objective:** Define Pydantic/dataclass models for memories, sessions, search results

**Files:**
- Create: `src/memory_agent/models.py`

### Task 3: Configuration

**Objective:** Config dataclass with all tunable parameters

**Files:**
- Create: `src/memory_agent/core/config.py`

### Task 4: Memory Store (SQLite)

**Objective:** Implement SQLite-backed persistent storage with CRUD for memories, embeddings, tags, sessions

**Files:**
- Create: `src/memory_agent/core/memory_store.py`
- Create: `tests/test_memory_store.py`

### Task 5: Embeddings Engine

**Objective:** sentence-transformers wrapper for encoding text into vectors, caching, batch processing

**Files:**
- Create: `src/memory_agent/core/embeddings.py`
- Create: `tests/test_embeddings.py`

### Task 6: Forgetting Mechanism

**Objective:** Ebbinghaus forgetting curve with importance-modulated decay, strength tracking, archival

**Files:**
- Create: `src/memory_agent/core/forgetting.py`
- Create: `tests/test_forgetting.py`

### Task 7: Retrieval Engine

**Objective:** Combined scoring (semantic + recency + importance + strength) with MMR diversity

**Files:**
- Create: `src/memory_agent/core/retrieval.py`
- Create: `tests/test_retrieval.py`

### Task 8: Agent Orchestrator

**Objective:** Full agent loop with perceive-retrieve-augment-decide-memorize-decay cycle

**Files:**
- Create: `src/memory_agent/agent/orchestrator.py`
- Create: `src/memory_agent/agent/decision.py`
- Create: `tests/test_agent.py`

### Task 9: CLI Interface

**Objective:** Click-based CLI for interactive multi-turn sessions

**Files:**
- Create: `src/memory_agent/cli/commands.py`
- Create: `src/memory_agent/__main__.py`

### Task 10: Demo & Examples

**Objective:** Runable demonstrations

**Files:**
- Create: `examples/demo_basic.py`
- Create: `examples/demo_multi_session.py`

---

## Execution Order

1. Task 1 (scaffold) → 2 (models) → 3 (config) → 4 (store) → 5 (embeddings)
2. Tasks 6 (forgetting) and 7 (retrieval) can parallel with 4/5 done
3. Task 8 (orchestrator) depends on 4-7
4. Task 9 (CLI) depends on 8
5. Task 10 (demo) is last
