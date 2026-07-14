"""Contract tests for MCP singleton session namespace isolation."""

from __future__ import annotations

from types import SimpleNamespace

from memory_agent.integrations import mcp_server


class _SessionAgent:
    """Minimal agent surface required by ``mcp_server._ensure_session``."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(session_id=None, namespace=None)
        self.init_calls: list[tuple[str, str | None]] = []
        self.end_calls = 0
        self.events: list[tuple[str, str | None]] = []

    def init_session(self, name: str, *, namespace: str | None = None) -> None:
        self.init_calls.append((name, namespace))
        self.events.append(("init", namespace))
        self.state.session_id = len(self.init_calls)
        self.state.namespace = namespace

    def end_session(self) -> None:
        self.end_calls += 1
        self.events.append(("end", self.state.namespace))
        self.state.session_id = None
        self.state.namespace = None


def test_ensure_session_switches_from_tenant_to_unscoped_without_duplicate_init(
    monkeypatch,
) -> None:
    """A namespace change ends the old session before starting the new one."""
    agent = _SessionAgent()
    monkeypatch.setattr(mcp_server, "_get_agent", lambda: agent)

    mcp_server._ensure_session(namespace="tenant-a")
    mcp_server._ensure_session(namespace="tenant-a")
    mcp_server._ensure_session(namespace=None)

    assert agent.init_calls == [("mcp-auto", "tenant-a"), ("mcp-auto", None)]
    assert agent.end_calls == 1
    assert agent.events == [
        ("init", "tenant-a"),
        ("end", "tenant-a"),
        ("init", None),
    ]
