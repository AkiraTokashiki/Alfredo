#!/usr/bin/env python
"""Basic demo: single-session memory agent interaction."""

import tempfile
from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent


def main():
    print("=" * 60)
    print("MemoryAgent — Demo Basico")
    print("=" * 60)

    # Use temp db so we start clean
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    agent = MemoryAgent(db_path=db_path)
    agent.init_session("demo-basico")

    interactions = [
        "Hola! Me llamo Manija",
        "Me gusta programar en Python",
        "Mi framework favorito es Next.js",
        "Trabajo como developer de Android",
        "Que sabes de mi?",
        "Cual es mi lenguaje favorito?",
        "Que framework me gusta?",
        "Donde trabajo?",
    ]

    for user_input in interactions:
        print(f"\n{'─' * 50}")
        print(f"  Tu: {user_input}")

        result = agent.perceive(user_input)

        # Show extracted memories
        if result["new_memories"]:
            print(f"  [{len(result['new_memories'])} nuevos recuerdos]")

        # Show recollections
        if result["recollections"]:
            print(f"  Recuerdos recuperados ({len(result['recollections'])}):")
            for r in result["recollections"][:3]:
                print(f"    [{r.memory.memory_type}] {r.memory.content[:60]}")

        if result["archived"]:
            print(f"  [{result['archived']} recuerdos archivados]")

    # Final stats
    print(f"\n{'=' * 60}")
    print("Estadisticas finales:")
    stats = agent.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")

    agent.close()
    Path(db_path).unlink(missing_ok=True)
    print(f"\nDemo completada.")


if __name__ == "__main__":
    main()
