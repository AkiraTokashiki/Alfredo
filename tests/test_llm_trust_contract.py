"""Contract test for the LLM connector's trust-aware memory boundary."""

from __future__ import annotations

from memory_agent.integrations.llm_connector import LLMConnector


class _ForbiddenRetrieval:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("LLM connector bypassed MemoryAgent.search_memories")


class _TrustAwareAgent:
    retrieval = _ForbiddenRetrieval()

    def search_memories(self, query: str, *, top_k: int = 5) -> dict:
        return {
            "results": [
                {
                    "id": 11,
                    "content": "The user prefers Python.",
                    "type": "preference",
                    "score": 0.91,
                },
                {
                    "id": 33,
                    "content": "The user disclosed an unclassified secret.",
                    "type": "semantic",
                    "score": 0.88,
                },
            ],
            "selected_ids": [11, 33],
            "dropped_ids": [22],
            "evidence": [
                {"id": 11, "trust": "trusted", "reason": "semantic match"},
                {"id": 33, "trust": "unknown", "reason": "unclassified"},
                {"id": 22, "trust": "untrusted", "reason": "dropped: stale"},
            ],
        }


def test_memory_context_uses_trust_aware_search_facade_without_raw_retrieval():
    """The LLM context must format selected facade results, not raw retrieval output."""
    connector = LLMConnector.__new__(LLMConnector)
    connector.agent = _TrustAwareAgent()

    context = connector._build_memory_context("Python preference")

    assert "The user prefers Python." in context
    assert "The user disclosed an unclassified secret." not in context
    assert "[preference]" in context
    assert "[MEMORIES]:" in context
    assert "[/MEMORIES]" in context
