# Alfredo Native Memory Vault Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store Alfredo's default SQLite memory database in a native Alfredo memory vault instead of whichever directory the command is run from.

**Architecture:** Add a small path resolver that prefers `ALFREDO_HOME`, then a development repo-local `.alfredo` directory, then the OS-native app data directory. Wire `MemoryAgentConfig` and the CLI through this resolver while keeping explicit `--db` and `db_path=` overrides intact.

**Tech Stack:** Python 3.11+, pathlib, os.environ, pytest, Click CLI.

---

## File map

- Create: `src/memory_agent/core/paths.py` — resolves the Alfredo home directory and default DB path.
- Modify: `src/memory_agent/core/config.py` — uses the native DB path as `MemoryAgentConfig.db_path` default.
- Modify: `src/memory_agent/cli/commands.py` — uses the config default unless `--db` is provided.
- Create: `tests/test_paths.py` — verifies env override, repo-local default, and config default behavior.
- Modify: `README.md` — documents `ALFREDO_HOME`, `.alfredo`, and the default DB location.

---

### Task 1: Add native path resolver

**Files:**
- Create: `src/memory_agent/core/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write failing path tests**

Create `tests/test_paths.py` with tests for:

```python
from __future__ import annotations

from pathlib import Path

from memory_agent.core.paths import default_memory_db_path, resolve_memory_home


def test_resolve_memory_home_uses_alfredo_home_env(monkeypatch, tmp_path):
    home = tmp_path / "custom-vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))

    resolved = resolve_memory_home(project_root=tmp_path / "repo")

    assert resolved == home
    assert resolved.exists()


def test_resolve_memory_home_uses_repo_local_alfredo_for_dev_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("ALFREDO_HOME", raising=False)
    repo = tmp_path / "Alfredo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "memory-agent"\n', encoding="utf-8")

    resolved = resolve_memory_home(project_root=repo)

    assert resolved == repo / ".alfredo"
    assert resolved.exists()


def test_default_memory_db_path_uses_memory_agent_db_filename(monkeypatch, tmp_path):
    home = tmp_path / "vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))

    assert default_memory_db_path().name == "memory_agent.db"
    assert default_memory_db_path().parent == home
```

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_agent.core.paths'`.

- [ ] **Step 2: Implement resolver**

Create `src/memory_agent/core/paths.py`:

```python
"""Native filesystem paths for Alfredo runtime state."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Alfredo"
DB_FILENAME = "memory_agent.db"


def _is_dev_repo(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        return 'name = "memory-agent"' in pyproject.read_text(encoding="utf-8")
    except OSError:
        return False


def _package_project_root() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    return root if _is_dev_repo(root) else None


def _os_app_data_home() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
    return Path.home() / ".alfredo"


def resolve_memory_home(project_root: Path | None = None) -> Path:
    """Return Alfredo's native memory directory and ensure it exists."""
    env_home = os.environ.get("ALFREDO_HOME")
    if env_home:
        home = Path(env_home).expanduser()
    else:
        root = project_root or _package_project_root()
        home = root / ".alfredo" if root and _is_dev_repo(root) else _os_app_data_home()

    home.mkdir(parents=True, exist_ok=True)
    return home


def default_memory_db_path() -> Path:
    """Return the default SQLite DB path for Alfredo memory."""
    return resolve_memory_home() / DB_FILENAME
```

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS.

---

### Task 2: Wire config and CLI defaults

**Files:**
- Modify: `src/memory_agent/core/config.py`
- Modify: `src/memory_agent/cli/commands.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Add config/CLI tests**

Append to `tests/test_paths.py`:

```python
from click.testing import CliRunner

from memory_agent.cli.commands import cli
from memory_agent.core.config import MemoryAgentConfig


def test_memory_agent_config_default_uses_native_db(monkeypatch, tmp_path):
    home = tmp_path / "vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))

    config = MemoryAgentConfig.default()

    assert Path(config.db_path) == home / "memory_agent.db"


def test_cli_stats_uses_native_default_db(monkeypatch, tmp_path):
    home = tmp_path / "vault"
    monkeypatch.setenv("ALFREDO_HOME", str(home))
    runner = CliRunner()

    result = runner.invoke(cli, ["stats"])

    assert result.exit_code == 0
    assert (home / "memory_agent.db").exists()
```

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL because `MemoryAgentConfig.db_path` still defaults to `memory_agent.db` and CLI still uses cwd when `--db` is omitted.

- [ ] **Step 2: Wire config**

Modify `src/memory_agent/core/config.py`:

```python
from memory_agent.core.paths import default_memory_db_path
```

Change the `MemoryAgentConfig` field:

```python
db_path: str = field(default_factory=lambda: str(default_memory_db_path()))
```

- [ ] **Step 3: Wire CLI**

Modify `src/memory_agent/cli/commands.py` DB path resolution:

```python
    config = MemoryAgentConfig.default()
    if db:
        db_path = Path(db).resolve()
    else:
        db_path = Path(config.db_path).expanduser().resolve()
```

Keep the existing `--db` override behavior.

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS.

---

### Task 3: Document and verify

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Under Quick Start after install, add:

```markdown
### Native memory vault

By default, Alfredo stores runtime memory in a native SQLite vault instead of the current working directory.

Development checkout default:

```text
.alfredo/memory_agent.db
```

Override with:

```bash
set ALFREDO_HOME=E:\code\alfredo\.alfredo   # Windows CMD
export ALFREDO_HOME="$PWD/.alfredo"          # Linux/macOS
```

You can still pass an explicit DB path:

```bash
python -m memory_agent --db path/to/memory_agent.db chat
```
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/test_paths.py tests/test_agent.py -v
```

Expected: PASS.

- [ ] **Step 3: Run full suite and demo**

Run:

```bash
python -m pytest tests -v
python examples/demo_hackathon.py
```

Expected: all tests pass; demo completes and creates/uses `.alfredo/memory_agent.db` by default.
