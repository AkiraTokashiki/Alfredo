"""Black-box contracts for the public Task4 documentation surface.

These checks intentionally read documentation as a user would: they validate
copyable local commands, discoverable lifecycle concepts, and policy coverage.
They do not contact remote services or execute documented commands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    "README.md",
    "INTEGRATION.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "docs/ARCHITECTURE.md",
)
COMMAND_FENCE = re.compile(
    r"```(?:bash|console|powershell|shell|sh|cmd|batch|zsh)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _required_doc(relative_path: str) -> str:
    """Load a public document while making an absent contract a test failure."""

    path = REPO_ROOT / relative_path
    if not path.is_file():
        pytest.fail(f"Task4 requires public documentation file: {relative_path}")
    return path.read_text(encoding="utf-8")


def _normalise(markdown: str) -> str:
    """Make prose/code matching tolerant of Markdown punctuation and wrapping."""

    return re.sub(r"[`*_]", "", markdown).replace("\u2013", "-").replace("\u2014", "-")


def _headings_and_sections(markdown: str) -> list[tuple[str, str]]:
    """Return each heading together with its content up to the next same-level heading."""

    headings = list(HEADING.finditer(markdown))
    sections: list[tuple[str, str]] = []
    for index, heading in enumerate(headings):
        level = len(heading.group(1))
        end = len(markdown)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        sections.append((heading.group(2), markdown[heading.start() : end]))
    return sections


def _recipe_section(markdown: str, client_pattern: str) -> str:
    """Find a client recipe section, with a bounded fallback for prose/table layouts."""

    sections = [
        body
        for heading, body in _headings_and_sections(markdown)
        if re.search(client_pattern, heading, re.IGNORECASE)
    ]
    if sections:
        return "\n".join(sections)

    match = re.search(client_pattern, markdown, re.IGNORECASE)
    if not match:
        pytest.fail(f"missing MCP client recipe: {client_pattern}")
    start = max(0, match.start() - 500)
    return markdown[start : match.end() + 900]


def _mentions_mcp_extra_or_absence(text: str) -> bool:
    """Require an actionable mcp-extra statement, not a bare mention of MCP."""

    text = _normalise(text).casefold()
    extra_relation = re.compile(
        r"(?:\[\s*mcp\s*\]|mcp.{0,100}extra|extra.{0,100}mcp|"
        r"mcp.{0,100}(?:optional|dependency|dependencia)|"
        r"(?:optional|dependency|dependencia).{0,100}mcp)",
        re.DOTALL,
    )
    absence_relation = re.compile(
        r"(?:no|not|without|does not require|doesn't require|sin|no necesita|"
        r"no requiere).{0,100}(?:mcp|extra)|(?:mcp|extra).{0,100}(?:not required|"
        r"not needed|no se necesita|no requerido)",
        re.DOTALL,
    )
    return bool(
        re.search(r"\bmcp\b", text)
        and (extra_relation.search(text) or absence_relation.search(text))
    )


def test_integration_publishes_local_install_compatibility_and_mcp_recipes() -> None:
    """Integration docs expose supported local entry points and all four MCP clients."""

    integration = _required_doc("INTEGRATION.md")
    commands = [
        line.strip()
        for block in COMMAND_FENCE.findall(integration)
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "python -m pip install alfredo-memory-agent" in commands
    assert "alfredo --offline quickstart" in commands
    assert re.search(r"\bpython\s+-m\s+memory_agent\b", integration)
    assert re.search(r"\bstdio\b", integration, re.IGNORECASE)
    assert re.search(r"\bHTTP\b", integration, re.IGNORECASE)

    clients = {
        "Hermes": r"\bHermes\b",
        "Claude Desktop": r"\bClaude\s+Desktop\b",
        "Cursor": r"\bCursor\b",
        "generic MCP client": (
            r"(?:generic\s+(?:MCP\s+)?clients?|any\s+MCP(?:-compatible)?\s+client|"
            r"cliente\s+(?:MCP\s+)?gen[eé]rico)"
        ),
    }
    for client, pattern in clients.items():
        recipe = _recipe_section(integration, pattern)
        assert re.search(r"\bstdio\b", recipe, re.IGNORECASE), client
        assert re.search(r"\bhttp\b", recipe, re.IGNORECASE), client
        assert _mentions_mcp_extra_or_absence(recipe), (
            f"{client} recipe must say whether the mcp extra is required"
        )


def test_architecture_documents_real_lifecycle_and_local_boundaries() -> None:
    """Architecture docs describe lifecycle decisions, guards, and benchmark/privacy limits."""

    architecture = _normalise(_required_doc("docs/ARCHITECTURE.md")).casefold()
    lifecycle_terms = {
        "perceive": r"\bperceive\w*\b|\bpercib\w*\b",
        "extract": r"\bextract\w*\b|\bextra(?:e|er|cción|ccion)\w*\b",
        "validate": r"\bvalidat\w*\b|\bvalid(?:a|ación|acion)\w*\b",
        "trust": r"\btrust\w*\b|\bconfian(?:za|ce)\w*\b",
        "store": r"\bstor(?:e|es|ed|age)\w*\b|\balmacen\w*\b",
        "retrieve": r"\bretriev\w*\b|\brecuper\w*\b",
        "context": r"\bcontext\w*\b|\bcontexto\w*\b",
        "reinforce": r"\breinforc\w*\b|\breforz\w*\b",
        "supersede": r"\bsupersed\w*\b|\breemplaz\w*\b|\bsustit\w*\b",
        "decay": r"\bdecay\w*\b|\bdeca(?:y|imient|e)\w*\b",
        "archive": r"\barchiv\w*\b",
    }
    for label, pattern in lifecycle_terms.items():
        assert re.search(pattern, architecture), f"missing lifecycle concept: {label}"

    assert re.search(r"\bsqlite\b", architecture)
    assert re.search(
        r"provider.{0,180}dimension|dimension.{0,180}provider", architecture, re.DOTALL
    )
    assert re.search(r"\b(?:guard|check|validat|mismatch|incompatib)", architecture)

    embedding_windows = [
        architecture[max(0, match.start() - 220) : match.end() + 220]
        for match in re.finditer(r"embedding", architecture)
    ]
    assert any(
        re.search(r"deterministic", window) and re.search(r"offline|local", window)
        for window in embedding_windows
    ), "architecture must bound embeddings to deterministic offline behavior"

    assert re.search(r"selected[ _-]?ids?", architecture)
    assert re.search(r"dropped[ _-]?ids?", architecture)
    assert re.search(r"\bsynthetic\b", architecture)
    assert re.search(r"\bprivacy\b", architecture)
    assert re.search(
        r"(?:synthetic|benchmark).{0,320}(?:not|does not|doesn't|isn't|cannot|"
        r"no substitute|not a substitute).{0,140}(?:privacy|security|audit)"
        r"|(?:privacy|security|audit).{0,320}(?:synthetic|benchmark).{0,180}"
        r"(?:not|does not|isn't|no substitute|not a substitute)",
        architecture,
        re.DOTALL,
    ), "synthetic benchmark evidence must be separated from production privacy controls"


def test_contributing_explains_reproducible_contributor_workflow() -> None:
    """Contributor guidance covers setup, tests, commits, and provider extension boundaries."""

    contributing = _normalise(_required_doc("CONTRIBUTING.md")).casefold()
    assert re.search(r"editable|editabl", contributing)
    assert re.search(r"(?:python\s+-m\s+)?pip\s+install\s+-e\s+\.", contributing)
    assert re.search(r"focused|focus", contributing)
    assert re.search(r"python\s+-m\s+pytest\s+[^\n]*tests/", contributing)
    assert re.search(
        r"(?:full|complete|entire).{0,120}(?:test|pytest)|"
        r"(?:test|pytest).{0,120}(?:full|complete|entire)",
        contributing,
    )
    assert re.search(r"readme", contributing)
    assert re.search(r"smoke|quickstart|command", contributing)
    assert re.search(r"commit", contributing)
    assert re.search(
        r"(?:provider|proveedor).{0,300}(?:duplicate|duplicat|without|avoid|"
        r"never|no ).{0,160}(?:lifecycle|cycle|ciclo)",
        contributing,
    )


def test_security_policy_covers_reporting_isolation_forget_and_limits() -> None:
    """Security guidance names operational controls without promising an audit."""

    security = _normalise(_required_doc("SECURITY.md")).casefold()
    assert re.search(r"report", security)
    assert re.search(r"report.{0,180}(?:security|vulnerab|incident|contact|maintainer)", security)
    assert re.search(r"namespace", security)
    assert re.search(r"namespace.{0,180}(?:isolat|tenant|scope|boundary)", security)
    assert re.search(r"\bforget\w*\b", security)
    assert re.search(r"forget\w*.{0,180}(?:delet|remov|archiv|explicit)", security)
    assert re.search(
        r"(?:sensitive|secret|personal|pii).{0,240}(?:limit|cannot|do not|avoid|"
        r"not designed|restriction|precauc)",
        security,
    )
    assert re.search(r"synthetic", security) and re.search(r"benchmark", security)
    assert re.search(
        r"(?:synthetic|benchmark).{0,220}(?:not|does not|isn't|cannot|no).{0,100}"
        r"(?:security\s+)?audit|(?:security\s+)?audit.{0,220}(?:synthetic|benchmark).{0,100}"
        r"(?:not|does not|isn't|cannot|no)",
        security,
        re.DOTALL,
    )


def test_community_documents_are_complete_and_scope_future_work() -> None:
    """Community policy, release history, and roadmap distinguish shipped from future work."""

    conduct = _normalise(_required_doc("CODE_OF_CONDUCT.md")).casefold()
    assert re.search(r"standard|covenant", conduct), (
        "CODE_OF_CONDUCT must identify a complete policy standard"
    )
    for concept in ("acceptable", "unacceptable", "enforcement", "report"):
        assert concept in conduct, f"CODE_OF_CONDUCT missing {concept} policy language"
    assert re.search(r"report.{0,180}(?:contact|email|maintainer|team|channel)", conduct)
    assert re.search(r"(?:warning|temporary|permanent|ban|consequence|action)", conduct)

    changelog = _normalise(_required_doc("CHANGELOG.md")).casefold()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r"^version\s*=\s*[\"']([^\"']+)", pyproject, re.MULTILINE)
    assert version_match, "project release version must be discoverable"
    current_version = version_match.group(1).casefold()
    assert current_version in changelog, "CHANGELOG must record the current release line"
    assert not re.search(
        r"^#+\s*(?:unreleased|next|future)\b",
        changelog,
        re.IGNORECASE | re.MULTILINE,
    )

    roadmap = _normalise(_required_doc("ROADMAP.md")).casefold()
    assert re.search(r"local.{0,100}sdk|sdk.{0,100}local", roadmap)
    future_sections = "\n".join(
        body
        for heading, body in _headings_and_sections(roadmap)
        if re.search(r"future|planned|later|next|post[- ]?release|not in current", heading)
    )
    assert future_sections, "ROADMAP must label future work separately"
    for future_item in ("dashboard", "hosting", "typescript"):
        assert future_item in future_sections, f"{future_item} must be marked as future work"


def test_executable_documentation_does_not_publish_rejected_install_names() -> None:
    """Copyable command blocks never install the historical bare distribution names."""

    prohibited = re.compile(
        r"\bpip\s+install\s+(?:['\"]?)(?:alfredo|memory-agent)(?:['\"]?)(?:\s|$)",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    for relative_path in PUBLIC_DOCS:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            continue
        markdown = path.read_text(encoding="utf-8")
        for block in COMMAND_FENCE.findall(markdown):
            flattened = re.sub(r"\s+", " ", block)
            if prohibited.search(flattened):
                offenders.append(relative_path)
    assert not offenders, f"rejected executable install command in: {sorted(set(offenders))}"
