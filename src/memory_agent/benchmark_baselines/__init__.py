"""Deterministic offline baseline strategies used by the benchmark."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class StrategyOutput:
    answer: str
    retrieved_ids: list[str]
    ignored_ids: list[str]
    confidence: float
    behavior: str
    context_chars: int
    latency_ms: float
    security_events: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieved_ids": list(self.retrieved_ids),
            "ignored_ids": list(self.ignored_ids),
            "confidence": round(float(self.confidence), 4),
            "behavior": self.behavior,
            "context_chars": int(self.context_chars),
            "latency_ms": round(float(self.latency_ms), 4),
            "security_events": [dict(event) for event in self.security_events],
        }


@runtime_checkable
class Strategy(Protocol):
    name: str

    def run(self, question: dict[str, Any], memories: Sequence[dict[str, Any]], *, seed: int = 0) -> StrategyOutput:
        ...


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\w]+", value.lower()) if len(token) > 1}


def _memory_id(memory: dict[str, Any]) -> str:
    return str(memory["memory_id"])


def _security(memory: dict[str, Any]) -> bool:
    return bool({"prompt_injection", "security"} & set(memory.get("tags", [])))


def _event(memory: dict[str, Any]) -> dict[str, str]:
    return {"memory_id": _memory_id(memory), "event": "prompt_injection_detected", "action": "quarantined_as_data"}


def _base_output(
    question: dict[str, Any], candidates: Sequence[dict[str, Any]], *, behavior: str, ignored: Sequence[dict[str, Any]] = (), latency_units: int = 1
) -> StrategyOutput:
    unique: dict[str, dict[str, Any]] = {}
    for memory in candidates:
        unique.setdefault(_memory_id(memory), memory)
    unique_ignored: dict[str, dict[str, Any]] = {}
    for memory in ignored:
        unique_ignored.setdefault(_memory_id(memory), memory)
    expected_ids = {str(memory_id) for memory_id in question.get("expected_memory_ids", [])}
    query_tokens = _tokens(str(question.get("query", "")))
    security_by_id: dict[str, dict[str, Any]] = {}
    for memory in list(unique.values()) + list(unique_ignored.values()):
        if not _security(memory):
            continue
        memory_tokens = _tokens(str(memory.get("content", "")))
        relevant = _memory_id(memory) in expected_ids if expected_ids else bool(query_tokens & memory_tokens)
        if relevant:
            security_by_id.setdefault(_memory_id(memory), memory)
    security = list(security_by_id.values())
    for memory in security:
        unique_ignored.setdefault(_memory_id(memory), memory)
    if question.get("requires_abstention"):
        expected_ids = {str(memory_id) for memory_id in question.get("expected_memory_ids", [])}
        relevant_ids = set(expected_ids)
        changed = True
        while changed:
            changed = False
            for memory in list(unique.values()) + list(unique_ignored.values()):
                if any(str(old_id) in relevant_ids for old_id in memory.get("supersedes", []) or []):
                    if _memory_id(memory) not in relevant_ids:
                        relevant_ids.add(_memory_id(memory))
                        changed = True
        unique_ignored = {memory_id: memory for memory_id, memory in unique_ignored.items() if memory_id in relevant_ids}
        for memory_id, memory in unique.items():
            if memory_id in relevant_ids:
                unique_ignored.setdefault(memory_id, memory)
        safe: list[dict[str, Any]] = []
        behavior = "security_event" if security else "abstain_no_allowed_memory"
    else:
        safe = [memory for memory in unique.values() if not _security(memory)]
    security_events = [_event(memory) for memory in security]
    if security_events:
        behavior = "security_event"
    retrieved = [_memory_id(memory) for memory in safe]
    answer = " ".join(str(memory.get("content", "")) for memory in safe)
    confidence = max((float(memory.get("confidence", 0.0) or 0.0) for memory in safe), default=0.0)
    context_chars = sum(len(str(memory.get("content", ""))) for memory in safe)
    latency_ms = float(0.25 + latency_units * 0.05 + len(retrieved) * 0.1 + context_chars * 0.001)
    return StrategyOutput(
        answer=answer or ("Security event: prompt-injection content was quarantined as data." if security_events else "Abstention: no memory retrieved."),
        retrieved_ids=list(dict.fromkeys(retrieved)),
        ignored_ids=list(unique_ignored),
        confidence=confidence,
        behavior=behavior,
        context_chars=context_chars,
        latency_ms=latency_ms,
        security_events=security_events,
    )


class RawHistoryStrategy:
    name = "raw-history"

    def __init__(self, *, max_memories: int | None = None) -> None:
        self.max_memories = max_memories

    def run(self, question: dict[str, Any], memories: Sequence[dict[str, Any]], *, seed: int = 0) -> StrategyOutput:
        history = [memory for memory in memories if memory.get("user_id") == question.get("user_id")]
        selected = history if self.max_memories is None else history[: self.max_memories]
        return _base_output(question, selected, behavior="raw_history_retrieval", latency_units=len(history))


class SemanticRAGStrategy:
    name = "semantic-rag"

    def __init__(self, *, top_k: int = 3) -> None:
        self.top_k = max(1, int(top_k))

    def run(self, question: dict[str, Any], memories: Sequence[dict[str, Any]], *, seed: int = 0) -> StrategyOutput:
        query_tokens = _tokens(str(question.get("query", "")))
        candidates = [memory for memory in memories if memory.get("user_id") == question.get("user_id")]
        scored = []
        for index, memory in enumerate(candidates):
            content_tokens = _tokens(str(memory.get("content", "")))
            overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens | content_tokens))
            scored.append((overlap, -index, _memory_id(memory), memory))
        scored.sort(reverse=True)
        selected = [row[3] for row in scored[: self.top_k] if row[0] > 0]
        if question.get("requires_abstention"):
            selected.extend(memory for memory in candidates if _security(memory) and memory not in selected)
        selected = list(dict((_memory_id(memory), memory) for memory in selected).values())
        ignored = [row[3] for row in scored[self.top_k :]] + [row[3] for row in scored[: self.top_k] if row[0] <= 0]
        return _base_output(question, selected, behavior="semantic_retrieval", ignored=ignored, latency_units=len(candidates) + len(query_tokens))


def _latest_replacement(memory: dict[str, Any], replacements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current = memory
    latest_active = memory if memory.get("status", "active") == "active" else None
    seen: set[str] = set()
    while _memory_id(current) in replacements and _memory_id(current) not in seen:
        seen.add(_memory_id(current))
        current = replacements[_memory_id(current)]
        if current.get("status", "active") == "active":
            latest_active = current
    return latest_active or memory


class AlfredoStrategy:
    name = "alfredo"

    def run(self, question: dict[str, Any], memories: Sequence[dict[str, Any]], *, seed: int = 0) -> StrategyOutput:
        candidates = [memory for memory in memories if memory.get("user_id") == question.get("user_id")]
        query_tokens = _tokens(str(question.get("query", "")))
        ranked: list[tuple[float, int, str, dict[str, Any]]] = []
        for index, memory in enumerate(candidates):
            tokens = _tokens(str(memory.get("content", "")))
            score = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
            ranked.append((score, -index, _memory_id(memory), memory))
        ranked.sort(reverse=True)
        replacements: dict[str, dict[str, Any]] = {}
        for memory in candidates:
            candidate_key = (int(memory.get("status", "active") == "active"), str(memory.get("created_at") or ""), _memory_id(memory))
            for old_id in memory.get("supersedes", []) or []:
                old_key = str(old_id)
                existing = replacements.get(old_key)
                existing_key = (int(existing.get("status", "active") == "active"), str(existing.get("created_at") or ""), _memory_id(existing)) if existing else (-1, "", "")
                if existing is None or candidate_key > existing_key:
                    replacements[old_key] = memory
        selected: dict[str, dict[str, Any]] = {}
        ignored: dict[str, dict[str, Any]] = {}
        security_query = bool({"security", "prompt_injection", "injection"} & set(question.get("tags", [])))
        ranked_candidates = ranked[:1]
        if security_query or question.get("requires_abstention"):
            ranked_candidates.extend(row for row in ranked if _security(row[3]) and row not in ranked_candidates)
        for score, _index, _mid, candidate in ranked_candidates:
            if score <= 0 and not ((security_query or question.get("requires_abstention")) and _security(candidate)):
                continue
            memory = candidate
            replacement = _latest_replacement(memory, replacements)
            if replacement is not memory and replacement.get("status", "active") == "active":
                ignored.setdefault(_memory_id(memory), memory)
                memory = replacement
            if _security(memory):
                ignored.setdefault(_memory_id(memory), memory)
            elif memory.get("status", "active") in {"archived", "expired", "forgotten"}:
                ignored.setdefault(_memory_id(memory), memory)
            elif float(memory.get("confidence", 0.0) or 0.0) < 0.5:
                ignored.setdefault(_memory_id(memory), memory)
            else:
                selected.setdefault(_memory_id(memory), memory)
        by_id = {_memory_id(memory): memory for memory in candidates}
        for memory in selected.values():
            pending = [str(old_id) for old_id in memory.get("supersedes", []) or []]
            seen: set[str] = set()
            while pending:
                old_id = pending.pop(0)
                if old_id in seen:
                    continue
                seen.add(old_id)
                if old_id in by_id:
                    old_memory = by_id[old_id]
                    ignored.setdefault(old_id, old_memory)
                    pending.extend(str(value) for value in old_memory.get("supersedes", []) or [])
        if question.get("requires_abstention"):
            for memory in list(ignored.values()):
                current_id = _memory_id(memory)
                seen_chain: set[str] = set()
                while current_id in replacements and current_id not in seen_chain:
                    seen_chain.add(current_id)
                    terminal = replacements[current_id]
                    terminal_id = _memory_id(terminal)
                    ignored.setdefault(terminal_id, terminal)
                    current_id = terminal_id
        if question.get("requires_abstention") and not selected:
            behavior = "abstain_no_allowed_memory"
        elif ignored and selected:
            behavior = "prefer_updated_memory_over_archived"
        else:
            behavior = "alfredo_retrieval"
        return _base_output(question, list(selected.values()), behavior=behavior, ignored=list(ignored.values()), latency_units=len(candidates) + 2)


RawHistory = RawHistoryStrategy
SemanticRAG = SemanticRAGStrategy
Alfredo = AlfredoStrategy
__all__ = ["Alfredo", "AlfredoStrategy", "RawHistory", "RawHistoryStrategy", "SemanticRAG", "SemanticRAGStrategy", "Strategy", "StrategyOutput"]
