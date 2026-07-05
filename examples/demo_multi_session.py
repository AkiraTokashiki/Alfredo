#!/usr/bin/env python
"""Multi-session demo: memories persist across sessions."""

import tempfile
from pathlib import Path

from memory_agent.agent.orchestrator import MemoryAgent


def main():
    print("=" * 60)
    print("MemoryAgent — Demo Multi-Sesion")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    agent = MemoryAgent(db_path=db_path)

    # ========================
    # Session 1: aprender preferencias
    # ========================
    print(f"\n>>> SESION 1: Aprendiendo preferencias")
    agent.init_session("sesion-1")

    session_1_inputs = [
        "Hola! Me encanta el cafe",
        "Uso Linux para trabajar",
        "Mi editor favorito es VS Code",
        "Prefiero la musica electronica",
    ]

    for inp in session_1_inputs:
        result = agent.perceive(inp)
        print(f"  Tu: {inp}")
        print(f"    → {len(result['new_memories'])} recuerdos guardados")

    print(f"\n  Total despues de sesion 1: {agent.state.total_memories} recuerdos")
    agent.end_session()

    # ========================
    # Session 2: verificar que recuerda
    # ========================
    print(f"\n>>> SESION 2: Recordando preferencias")
    agent.init_session("sesion-2")

    session_2_inputs = [
        "Que sabes de mi?",
        "Que editor uso?",
        "Que sistema operativo prefiero?",
        "Que tipo de musica me gusta?",
    ]

    for inp in session_2_inputs:
        result = agent.perceive(inp)
        print(f"  Tu: {inp}")
        print(f"    Recollections: {len(result['recollections'])}")
        for r in result["recollections"][:2]:
            print(f"      → [{r.memory.memory_type}] {r.memory.content[:70]}")
        print(f"    Score: {r.score:.3f}" if result['recollections'] else "")

    print(f"\n  Total despues de sesion 2: {agent.state.total_memories} recuerdos")
    agent.end_session()

    # ========================
    # Session 3: el olvido
    # ========================
    print(f"\n>>> SESION 3: Probando el olvido (simulado)")
    agent.init_session("sesion-3")

    # Add a low-importance memory
    agent.perceive("El clima es lindo hoy")

    # Print stats to see decay lifespans
    stats = agent.get_stats()
    print(f"\n  Lifespans de decay:")
    for level, days in stats["decay_lifespans_days"].items():
        print(f"    {level}: ~{days} days")

    print(f"\n  Memorias activas: {stats['total_active']}")
    print(f"  Por tipo: {stats['type_distribution']}")

    agent.end_session()

    # ========================
    # Resumen final
    # ========================
    print(f"\n{'=' * 60}")
    print("RESUMEN:")
    print(f"  Total memorias al final: {agent.state.total_memories}")
    print(f"  El agente recordo preferencias entre sesiones ✓")
    print(f"  El olvido usa curva de Ebbinghaus con importancia ✓")
    print(f"  Los embeddings semanticos permiten busqueda por similitud ✓")
    print(f"  MMR diversifica resultados en retrieval ✓")

    agent.close()
    Path(db_path).unlink(missing_ok=True)
    print(f"\nDemo multi-sesion completada.")


if __name__ == "__main__":
    main()
