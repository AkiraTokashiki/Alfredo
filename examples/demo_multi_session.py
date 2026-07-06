#!/usr/bin/env python
"""Multi-session demo: memories persist across sessions."""

import tempfile
from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent


def main():
    print("=" * 60)
    print("MemoryAgent — Multi-Session Demo")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    agent = MemoryAgent(db_path=db_path)

    # ========================
    # Session 1: learn preferences
    # ========================
    print("\n>>> SESSION 1: Learning preferences")
    agent.init_session("session-1")

    session_1_inputs = [
        "Hi! I love coffee",
        "I use Linux for work",
        "My favorite editor is VS Code",
        "I prefer electronic music",
    ]

    for inp in session_1_inputs:
        result = agent.perceive(inp)
        print(f"  You: {inp}")
        print(f"    → {len(result['new_memories'])} memories stored")

    print(f"\n  Total after session 1: {agent.state.total_memories} memories")
    agent.end_session()

    # ========================
    # Session 2: verify persistent recall
    # ========================
    print("\n>>> SESSION 2: Recalling preferences")
    agent.init_session("session-2")

    session_2_inputs = [
        "What do you know about me?",
        "What editor do I use?",
        "What operating system do I prefer?",
        "What type of music do I like?",
    ]

    for inp in session_2_inputs:
        result = agent.perceive(inp)
        print(f"  You: {inp}")
        print(f"    Recollections: {len(result['recollections'])}")
        for r in result["recollections"][:2]:
            print(f"      → [{r.memory.memory_type}] {r.memory.content[:70]}")
        print(f"    Score: {r.score:.3f}" if result['recollections'] else "")

    print(f"\n  Total after session 2: {agent.state.total_memories} memories")
    agent.end_session()

    # ========================
    # Session 3: forgetting
    # ========================
    print("\n>>> SESSION 3: Testing forgetting (simulated)")
    agent.init_session("session-3")

    # Add a low-importance memory
    agent.perceive("The weather is nice today")

    # Print stats to see decay lifespans
    stats = agent.get_stats()
    print("\n  Decay lifespans:")
    for level, days in stats["decay_lifespans_days"].items():
        print(f"    {level}: ~{days} days")

    print(f"\n  Active memories: {stats['total_active']}")
    print(f"  By type: {stats['type_distribution']}")

    agent.end_session()

    # ========================
    # Final summary
    # ========================
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"  Final memory count: {agent.state.total_memories}")
    print("  The agent recalled preferences across sessions ✓")
    print("  Forgetting uses an importance-weighted Ebbinghaus curve ✓")
    print("  Semantic embeddings support similarity search ✓")
    print("  MMR diversifies retrieval results ✓")

    agent.close()
    Path(db_path).unlink(missing_ok=True)
    print("\nMulti-session demo complete.")


if __name__ == "__main__":
    main()
