#!/usr/bin/env python
"""No-voice Devpost video demo for Alfredo MemoryAgent.

Run this in a maximized terminal with a large font, then record the screen.
Use --fast for automated checks.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.benchmark import (
    evaluate_questions,
    load_memories_jsonl,
    load_questions_jsonl,
    load_users,
    seed_memory_store,
    write_report,
)


BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "alfredos_vault"
DEMO_WIDTH = 72
CAPTION_PAUSE_SECONDS = 5.0
SECTION_PAUSE_SECONDS = 3.0
TURN_BEFORE_RESULT_SECONDS = 1.5
TURN_AFTER_RESULT_SECONDS = 10.0
STATS_PAUSE_SECONDS = 6.0
FINAL_PAUSE_SECONDS = 7.0


def planned_pause_seconds() -> float:
    """Estimated wait time for the normal recording path."""
    captions = 2 * CAPTION_PAUSE_SECONDS
    sections = 4 * SECTION_PAUSE_SECONDS
    conversation_turns = 3 * (TURN_BEFORE_RESULT_SECONDS + TURN_AFTER_RESULT_SECONDS)
    benchmark_pauses = (
        TURN_AFTER_RESULT_SECONDS / 2
        + STATS_PAUSE_SECONDS
        + 5 * (TURN_AFTER_RESULT_SECONDS / 3)
    )
    return captions + sections + conversation_turns + benchmark_pauses + FINAL_PAUSE_SECONDS + 1.0


def pause(seconds: float, fast: bool) -> None:
    if not fast:
        time.sleep(seconds)


def rule(char: str = "=") -> str:
    return char * DEMO_WIDTH


def clear_screen(fast: bool) -> None:
    if fast:
        return
    print("\033[2J\033[H", end="")


def center(text: str) -> str:
    return text.center(DEMO_WIDTH)


def caption(title: str, lines: list[str], *, fast: bool) -> None:
    clear_screen(fast)
    print("\n" + rule("="))
    print(center(title.upper()))
    print(rule("="))
    print()
    for line in lines:
        print(center(line))
    print()
    print(rule("="))
    print()
    pause(CAPTION_PAUSE_SECONDS, fast)


def section(title: str, subtitle: str, *, fast: bool) -> None:
    clear_screen(fast)
    print("\n" + rule("#"))
    print(center(title.upper()))
    print(center(subtitle))
    print(rule("#"))
    print()
    pause(SECTION_PAUSE_SECONDS, fast)


def print_recollections(result: dict) -> None:
    recollection_text = result["recollection_text"].strip()
    print(recollection_text or "[no recollections yet]")
    print()
    print(f"active memories: {result['total_memories']}")
    print(f"archived this turn: {result['archived']}")
    packet = result.get("recall_packet")
    if packet is not None:
        print(f"context budget: {packet.used_chars}/{packet.available_chars} chars")
        print(f"omitted memories: {len(packet.omitted)}")


def show_turn(agent: MemoryAgent, user_text: str, *, fast: bool) -> dict:
    print(f">>> USER: {user_text}")
    print()
    pause(TURN_BEFORE_RESULT_SECONDS, fast)
    result = agent.perceive(user_text)
    print_recollections(result)
    print()
    pause(TURN_AFTER_RESULT_SECONDS, fast)
    return result

def compact_answer(text: str, limit: int = 170) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def print_benchmark_decision(title: str, result: dict) -> None:
    print(rule("-"))
    print(title.upper())
    print(rule("-"))
    print(f"query: {result['query']}")
    print(f"behavior: {result['behavior_detected']}")
    print(f"retrieved: {', '.join(result['retrieved_memory_ids']) or 'none'}")
    print(f"ignored: {', '.join(result['ignored_memory_ids']) or 'none'}")
    print(f"answer: {compact_answer(result['answer'])}")
    print()


def run_vault_benchmark(agent: MemoryAgent) -> tuple[dict, list[dict]]:
    users = load_users(BENCHMARK_DIR / "users.json", expected_count=25)
    memories = load_memories_jsonl(
        BENCHMARK_DIR / "memories.jsonl", expected_count=5000
    )
    questions = load_questions_jsonl(
        BENCHMARK_DIR / "evaluation_questions.jsonl", expected_count=500
    )
    seed_memory_store(agent.store, users, memories)
    results = evaluate_questions(agent.store, questions, users=users)
    report = write_report(results, BENCHMARK_DIR / "demo_evaluation_report.json")
    return report, results



def run_demo(*, fast: bool) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
        db_path = Path(db.name)

    agent = MemoryAgent(db_path=db_path)
    try:
        caption(
            "ALFREDO MEMORYAGENT",
            [
                "NO VOICEOVER DEMO",
                "Persistent memory for Qwen Cloud agents",
                "Remember. Retrieve. Forget stale context.",
            ],
            fast=fast,
        )

        section(
            "Session 1",
            "Alfredo learns a user preference",
            fast=fast,
        )
        agent.init_session("video-session-1")
        show_turn(agent, "I like Python and I prefer concise answers", fast=fast)
        agent.end_session()

        section(
            "Session 2",
            "A new session recalls the old preference",
            fast=fast,
        )
        agent.init_session("video-session-2")
        show_turn(agent, "What language do I like?", fast=fast)
        agent.end_session()

        section(
            "Session 3",
            "Alfredo updates stale memory",
            fast=fast,
        )
        agent.init_session("video-session-3")
        show_turn(agent, "I do not like Python", fast=fast)
        agent.end_session()

        caption(
            "LARGE VAULT, SMALL PROMPT",
            [
                "SQLite keeps the long-term vault.",
                "The model receives only a small trusted recall packet.",
                "Budget limits prompt injection, not memory capacity.",
            ],
            fast=fast,
        )

        section(
            "Vault Benchmark",
            "25 users, 5,000 memories, 500 evaluation questions",
            fast=fast,
        )
        print("ALFREDO VAULT BENCHMARK")
        print(rule("-"))
        print("users loaded: 25")
        print("memories loaded: 5,000")
        print("evaluation questions: 500")
        print("simulated continuity: 90 days")
        print("domains: health, work, code, agenda, documents, conversations")
        print()
        pause(TURN_AFTER_RESULT_SECONDS / 2, fast)

        report, results = run_vault_benchmark(agent)
        metrics = report["metrics"]
        print("BENCHMARK RESULTS")
        print(rule("-"))
        print(f"questions: {metrics['total_questions']}")
        print(f"passed: {metrics['passed']}")
        print(f"failed: {metrics['failed']}")
        print(f"accuracy: {metrics['accuracy_percentage']:.2f}%")
        print(f"security events: {metrics['security_events']}")
        print()
        pause(STATS_PAUSE_SECONDS, fast)

        by_question = {result["question_id"]: result for result in results}
        for title, question_id in [
            ("Temporal recall", "eval_user_001_010"),
            ("Contradiction update", "eval_user_003_005"),
            ("Expired memory filtered", "eval_user_004_011"),
            ("Low-confidence abstention", "eval_user_006_013"),
            ("Prompt injection resisted", "eval_user_007_014"),
        ]:
            print_benchmark_decision(title, by_question[question_id])
            pause(TURN_AFTER_RESULT_SECONDS / 3, fast)

        clear_screen(fast)
        print("\n" + rule("="))
        print(center("DEVPOST TAKEAWAY"))
        print(rule("="))
        print()
        for line in [
            "Alfredo is a benchmarked memory vault for agents.",
            "Large SQLite vault. Small trusted recall packet.",
            "CLI, MCP, and Qwen Cloud ready.",
        ]:
            print(center(line))
        print()
        print(rule("="))
        print()
        pause(FINAL_PAUSE_SECONDS, fast)
        print(
            center(
                f"Estimated recording time: about {planned_pause_seconds() / 60:.1f} minutes"
            )
        )
        print()
        pause(1.0, fast)
    finally:
        agent.close()
        db_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip pauses and screen clears for automated tests.",
    )
    args = parser.parse_args()

    columns = shutil.get_terminal_size((DEMO_WIDTH, 24)).columns
    if not args.fast and columns < DEMO_WIDTH:
        print(f"Tip: widen the terminal to at least {DEMO_WIDTH} columns.")
        pause(2.0, args.fast)

    run_demo(fast=args.fast)


if __name__ == "__main__":
    main()
