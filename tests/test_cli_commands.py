"""Contract tests for namespace-aware CLI adapter commands."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from memory_agent.cli import commands as commands_module
from memory_agent.cli.commands import cli


@dataclass
class _Call:
    name: str
    kwargs: dict


class _FacadeSpy:
    """Public facade fake; implementation-layer access is an immediate failure."""

    def __init__(self) -> None:
        self.calls: list[_Call] = []

    @property
    def retrieval(self):  # pragma: no cover - reached only on an invalid adapter
        raise AssertionError("CLI adapter bypassed MemoryAgent.search_memories")

    @property
    def store(self):  # pragma: no cover - reached only on an invalid adapter
        raise AssertionError("CLI adapter bypassed MemoryAgent facade for storage")

    @property
    def forgetting(self):  # pragma: no cover - reached only on an invalid adapter
        raise AssertionError("CLI adapter bypassed MemoryAgent facade for forgetting")

    def get_stats(self, *, namespace: str | None = None) -> dict:
        self.calls.append(_Call("get_stats", {"namespace": namespace}))
        return {
            "namespace": namespace,
            "total_active": 2,
            "archived": 1,
            "embedding_count": 2,
            "avg_importance": 0.75,
            "trust": "trusted",
            "reason": "active namespace summary",
            "lifecycle": "active",
            "type_distribution": {"preference": 2},
            "decay_lifespans_days": {"high": 30},
        }

    def search_memories(
        self,
        query: str,
        *,
        top_k: int = 5,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> dict:
        self.calls.append(
            _Call(
                "search_memories",
                {
                    "query": query,
                    "top_k": top_k,
                    "memory_type": memory_type,
                    "namespace": namespace,
                },
            )
        )
        return {
            "namespace": namespace,
            "results": [
                {
                    "id": 11,
                    "content": "The user prefers Python.",
                    "type": "preference",
                    "score": 0.91,
                    "trust": "trusted",
                    "reason": "semantic match",
                }
            ],
            "selected_ids": [11],
            "dropped_ids": [22],
            "evidence": [
                {"id": 11, "trust": "trusted", "reason": "semantic match"},
                {"id": 22, "trust": "untrusted", "reason": "dropped: stale"},
            ],
            "lifecycle": {"status": "searched", "namespace": namespace},
        }

    def forget_memory(
        self, memory_id: int, *, namespace: str | None = None
    ) -> dict:
        self.calls.append(
            _Call(
                "forget_memory",
                {"memory_id": memory_id, "namespace": namespace},
            )
        )
        return {
            "id": memory_id,
            "namespace": namespace,
            "status": "archived",
            "trust": "trusted",
            "reason": "explicit user request",
            "lifecycle": "archived",
        }


    def list_memories(self, *, namespace: str | None = None) -> list:
        self.calls.append(_Call("list_memories", {"namespace": namespace}))
        return []
    def explain_memory(self, memory) -> dict:
        return {
            "trust": "trusted" if memory.confidence >= 0.5 else "untrusted",
            "reason": "configured trust policy",
        }

@pytest.fixture
def facade(monkeypatch: pytest.MonkeyPatch) -> _FacadeSpy:
    fake = _FacadeSpy()
    monkeypatch.setattr(commands_module, "_get_agent", lambda _ctx: fake)
    return fake


def test_stats_accepts_namespace_and_displays_lifecycle_trust_reason(
    facade: _FacadeSpy,
) -> None:
    result = CliRunner().invoke(cli, ["stats", "--namespace", "tenant-a"])

    assert result.exit_code == 0, result.output
    assert facade.calls == [_Call("get_stats", {"namespace": "tenant-a"})]
    output = result.output.lower()
    assert "tenant-a" in output
    assert "trusted" in output
    assert "active namespace summary" in output
    assert "lifecycle" in output


def test_search_accepts_namespace_and_uses_facade_without_retrieval_bypass(
    facade: _FacadeSpy,
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "search",
            "preferred language",
            "--namespace",
            "tenant-a",
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert facade.calls == [
        _Call(
            "search_memories",
            {
                "query": "preferred language",
                "top_k": 1,
                "memory_type": None,
                "namespace": "tenant-a",
            },
        )
    ]
    output = result.output.lower()
    assert "tenant-a" in output
    assert "trusted" in output
    assert "semantic match" in output
    assert "lifecycle" in output
    assert "11" in output and "22" in output


def test_forget_command_accepts_namespace_and_reports_lifecycle_reason(
    facade: _FacadeSpy,
) -> None:
    result = CliRunner().invoke(
        cli, ["forget", "11", "--namespace", "tenant-a"]
    )

    assert result.exit_code == 0, result.output
    assert facade.calls == [
        _Call(
            "forget_memory",
            {"memory_id": 11, "namespace": "tenant-a"},
        )
    ]
    output = result.output.lower()
    assert "tenant-a" in output
    assert "archived" in output
    assert "trusted" in output
    assert "explicit user request" in output
    assert "lifecycle" in output
def test_slash_commands_use_public_facade_without_storage_bypass(
    facade: _FacadeSpy, capsys: pytest.CaptureFixture[str]
) -> None:
    commands_module._handle_command(facade, "/memories")
    commands_module._handle_command(facade, "/search preferred language")
    commands_module._handle_command(facade, "/forget 11")

    assert [call.name for call in facade.calls] == [
        "list_memories",
        "search_memories",
        "forget_memory",
    ]
    output = capsys.readouterr().out.lower()
    assert "no memories" in output
    assert "search results" in output
    assert "archived" in output
    assert "namespace" in output
    assert "selected ids" in output
    assert "dropped ids" in output
    assert "trusted" in output
    assert "semantic match" in output
    assert "explicit user request" in output
def test_slash_search_empty_still_reports_metadata(
    facade: _FacadeSpy, capsys: pytest.CaptureFixture[str]
) -> None:
    def empty_search(query: str, *, top_k: int = 5, memory_type=None, namespace=None):
        return {
            "namespace": "tenant-empty",
            "results": [],
            "selected_ids": [],
            "dropped_ids": [],
            "evidence": [],
            "lifecycle": {"status": "searched", "namespace": "tenant-empty"},
        }

    facade.search_memories = empty_search
    commands_module._handle_command(facade, "/search missing")

    output = capsys.readouterr().out.lower()
    assert "no results" in output
    assert "tenant-empty" in output
    assert "lifecycle" in output
    assert "selected ids" in output
    assert "dropped ids" in output
def test_slash_memories_reports_namespace_lifecycle_and_evidence(
    facade: _FacadeSpy, capsys: pytest.CaptureFixture[str]
) -> None:
    facade.namespace = "tenant-a"
    facade.list_memories = lambda *, namespace=None: [
        SimpleNamespace(
            id=7,
            content="Python preference",
            memory_type="preference",
            importance=0.8,
            strength=1.0,
            access_count=1,
            tags=[],
            confidence=0.9,
            last_decision_reason="active preference",
        )
    ]
    commands_module._handle_command(facade, "/memories")

    output = capsys.readouterr().out.lower()
    assert "tenant-a" in output
    assert "lifecycle" in output
    assert "selected ids" in output
    assert "evidence #7" in output
    assert "trusted" in output
    assert "configured trust policy" in output
