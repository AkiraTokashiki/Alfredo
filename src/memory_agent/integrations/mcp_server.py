"""MCP Server for MemoryAgent.

Exposes MemoryAgent as MCP tools so any MCP client (Hermes, Claude Desktop, Cursor)
can use its persistent memory capabilities.

Usage:
    python -m memory_agent mcp                  # stdio transport (for Hermes config)
    python -m memory_agent mcp --http 8080      # HTTP transport (for remote clients)

Hermes config to add:
    mcp_servers:
      memory-agent:
        command: python
        args: ["-m", "memory_agent", "mcp"]

DB path: set MEMORY_AGENT_DB env var, or defaults to ./memory_agent.db
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig
from memory_agent.models import MemoryRecord

# ---------------------------------------------------------------------------
# Singleton agent with thread safety
# ---------------------------------------------------------------------------

_agent: MemoryAgent | None = None
_agent_lock = threading.Lock()


def _resolve_db_path() -> Path:
    """Resolve DB path: env var MEMORY_AGENT_DB, or cwd/memory_agent.db."""
    env_path = os.environ.get("MEMORY_AGENT_DB")
    if env_path:
        return Path(env_path).resolve()
    return Path.cwd() / "memory_agent.db"


def _get_agent() -> MemoryAgent:
    """Lazy init with thread lock so the model is only loaded once."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:  # double-check after acquiring lock
                config = MemoryAgentConfig.default()
                db_path = _resolve_db_path()
                _agent = MemoryAgent(config=config, db_path=db_path)
    return _agent


def _ensure_session() -> None:
    """Auto-start a session if none active."""
    agent = _get_agent()
    if agent.state.session_id is None:
        agent.init_session("mcp-auto")


# Create MCP server
mcp = FastMCP(
    "MemoryAgent",
    instructions="""MemoryAgent — persistent memory system for AI agents.

Provides tools to store and retrieve memories across sessions with:
- Semantic search (sentence-transformers embeddings)
- Ebbinghaus forgetting curve (memories decay over time unless reinforced)
- Multi-type memory (episodic, semantic, preferences, procedures)
- MMR diversity to avoid near-duplicate results
""",
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="memory__perceive",
    description="Process user input: extract, store, and retrieve memories in one step. "
    "Returns recollections relevant to the input plus any new memories extracted.",
)
async def memory_perceive(
    user_input: str,
    top_k: int = 5,
) -> str:
    """Process a user input through the full memory cycle.

    Args:
        user_input: The user's message or query.
        top_k: Max recollections to return (default: 5).

    Returns:
        JSON with recollections, new_memories, and stats.
    """
    _ensure_session()
    agent = _get_agent()
    result = agent.perceive(user_input)
    recollections = result["recollections"][:top_k]

    output = {
        "recollections": [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "type": r.memory.memory_type,
                "importance": r.memory.importance,
                "strength": r.memory.strength,
                "score": round(r.score, 3),
            }
            for r in recollections
        ],
        "new_memories": [
            {
                "id": m.id,
                "content": m.content[:80],
                "type": m.memory_type,
            }
            for m in result["new_memories"]
        ],
        "stats": {
            "turn": result["turn_count"],
            "total_memories": result["total_memories"],
            "archived": result["archived"],
        },
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool(
    name="memory__search",
    description="Semantic search across all stored memories. "
    "Finds relevant memories even without exact keyword matches.",
)
async def memory_search(
    query: str,
    top_k: int = 5,
    memory_type: str | None = None,
) -> str:
    """Search memories semantically.

    Args:
        query: Natural language query.
        top_k: Max results (default: 5).
        memory_type: Optional filter: episodic, semantic, preference, procedural.

    Returns:
        JSON array of matching memories with scores.
    """
    _ensure_session()
    agent = _get_agent()
    results = agent.retrieval.retrieve(
        query, top_k=top_k, memory_type=memory_type, use_mmr=True
    )

    if not results:
        return json.dumps({"results": [], "total": 0})

    output = {
        "results": [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "type": r.memory.memory_type,
                "importance": r.memory.importance,
                "strength": r.memory.strength,
                "access_count": r.memory.access_count,
                "tags": r.memory.tags,
                "score": round(r.score, 3),
                "scores": {
                    "semantic": round(r.semantic_score, 3),
                    "recency": round(r.recency_score, 3),
                    "importance": round(r.importance_score, 3),
                    "strength": round(r.strength_score, 3),
                },
            }
            for r in results
        ],
        "total": len(results),
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool(
    name="memory__store",
    description="Manually store a memory. Use when you want to explicitly "
    "save a fact, preference, or experience for future recall.",
)
async def memory_store(
    content: str,
    memory_type: str = "semantic",
    importance: float = 0.5,
    tags: str = "",
) -> str:
    """Store a new memory.

    Args:
        content: The memory content text.
        memory_type: episodic, semantic, preference, or procedural.
        importance: [0-1] how important this memory is (higher = decays slower).
        tags: Comma-separated tags for categorization.

    Returns:
        JSON with the new memory id.
    """
    _ensure_session()
    agent = _get_agent()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Clamp importance to valid range
    importance = max(0.0, min(1.0, importance))

    mem = MemoryRecord(
        content=content,
        memory_type=memory_type,
        importance=importance,
        tags=tag_list,
    )
    mid = agent.store_memory(mem)

    return json.dumps(
        {
            "id": mid,
            "content": content[:60],
            "type": memory_type,
            "importance": importance,
            "status": "stored",
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="memory__stats",
    description="Get memory agent statistics: total memories, type distribution, "
    "decay lifespans, embedding count, and archival info.",
)
async def memory_stats() -> str:
    """Return memory statistics."""
    _ensure_session()
    agent = _get_agent()
    stats = agent.get_stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


@mcp.tool(
    name="memory__forget",
    description="Delete a specific memory by ID. Use when a memory is wrong or obsolete.",
)
async def memory_forget(memory_id: int) -> str:
    """Delete a memory.

    Args:
        memory_id: The memory ID to delete.

    Returns:
        Confirmation message.
    """
    agent = _get_agent()
    agent.store.delete_memory(memory_id)
    return json.dumps({"id": memory_id, "status": "deleted"})


@mcp.tool(
    name="memory__reinforce",
    description="Reinforce a memory by ID — boosts its recall strength. "
    "Use when the user confirms a memory is still relevant.",
)
async def memory_reinforce(memory_id: int) -> str:
    """Reinforce a memory (boost recall strength).

    Args:
        memory_id: The memory ID to reinforce.

    Returns:
        New strength value.
    """
    agent = _get_agent()
    mem = agent.store.get_memory(memory_id)
    if mem is None:
        return json.dumps({"error": f"Memory #{memory_id} not found"})

    agent.forgetting.reinforce(mem)
    agent.store.update_memory(mem)
    return json.dumps(
        {
            "id": memory_id,
            "new_strength": round(mem.strength, 3),
            "status": "reinforced",
        }
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource(
    uri="memory://recent",
    name="Recent Memories",
    description="The 10 most recent active memories.",
    mime_type="application/json",
)
async def recent_memories() -> str:
    """Return the 10 most recent active memories."""
    agent = _get_agent()
    memories = agent.store.get_all_active_memories()
    recent = memories[:10]
    output = [
        {
            "id": m.id,
            "content": m.content[:80],
            "type": m.memory_type,
            "importance": m.importance,
            "strength": round(m.strength, 3),
            "created_at": m.created_at,
        }
        for m in recent
    ]
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.resource(
    uri="memory://stats",
    name="Memory Stats",
    description="Current memory agent statistics.",
    mime_type="application/json",
)
async def stats_resource() -> str:
    """Return memory stats as a resource."""
    agent = _get_agent()
    stats = agent.get_stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="memory-assisted",
    description="Load relevant memories before responding to a user query.",
)
async def memory_assisted_prompt(query: str) -> str:
    """Generate a prompt that includes relevant memories.

    Args:
        query: The user's question or request.
    """
    agent = _get_agent()
    results = agent.retrieval.retrieve(query, top_k=5, use_mmr=True)

    if not results:
        return f"## Query\n\n{query}\n\n*(No relevant memories found)*"

    memories_text = "\n".join(
        f"- [{r.memory.memory_type}] (imp={r.memory.importance:.1f}) {r.memory.content}"
        for r in results
    )

    return f"""## Relevant Memories
{memories_text}

## Query
{query}

## Instruction
Using the relevant memories above to inform your response, answer the user's query.
If memories suggest a preference or past experience, reference it naturally."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_mcp_server(host: str | None = None, port: int | None = None) -> None:
    """Start the MCP server.

    Args:
        host: HTTP host (HTTP mode). None = stdio mode.
        port: HTTP port (HTTP mode). None = stdio mode.
    """
    # Pre-warm agent so first MCP call isn't slow
    _get_agent()

    if host and port:
        print(f"MemoryAgent MCP Server at http://{host}:{port}/mcp", file=sys.stderr)
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
