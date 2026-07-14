# Alfredo Adoption and Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Alfredo an immediately understandable and installable open-source MemoryAgent while preserving its core identity, lifecycle behavior, benchmark evidence, and local-first architecture.

**Architecture:** Keep `Alfredo` as the product brand and `memory_agent` as the Python import/module. Publish a collision-free PyPI distribution named `alfredo-memory-agent`; do not use `alfredo` or `memory-agent`, which already belong to unrelated PyPI projects. Add a lightweight offline-first CLI entry point, then align README, demos, integrations, CI, release metadata, and community documentation around the same public contract.

**Tech Stack:** Python 3.11+, setuptools/wheel, Click, SQLite, deterministic embeddings, sentence-transformers optional semantic provider, pytest, GitHub Actions, PyPI trusted publishing, Markdown, SVG.

---

## File map before implementation

- Modify `pyproject.toml`: distribution name, public description, URLs, classifiers, optional dependencies, console script, and package metadata.
- Modify `src/memory_agent/cli/commands.py`: branded CLI help/output and a deterministic first-run path that remains compatible with `python -m memory_agent`.
- Modify `src/memory_agent/__init__.py`: expose a single version constant if packaging metadata requires it; retain existing public exports.
- Create `tests/test_package_metadata.py`: verify distribution metadata, console entry point, and import/module compatibility.
- Modify `tests/test_cli_quickstart.py` and `tests/test_documentation_commands.py`: verify the branded quickstart and documented commands.
- Modify `README.md`: replace the current hackathon-first opening with an adoption-oriented MemoryAgent hero, two-command quickstart, visual lifecycle, honest comparison, benchmark proof, integrations, and contribution path.
- Create `docs/assets/alfredo-memory-lifecycle.svg`: lightweight visual showing learn → retrieve → trust → pack → reinforce → supersede/forget.
- Modify `examples/demo_basic.py` and `examples/demo_hackathon.py`: deterministic, branded output and a short lifecycle path suitable for README recording.
- Create `examples/demo_lifecycle.py`: a non-interactive, offline, cross-session lifecycle demo with stable output.
- Modify `INTEGRATION.md` and `docs/ARCHITECTURE.md`: align names, installation, MCP recipes, public lifecycle, and security/privacy boundaries.
- Create `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, and `ROADMAP.md`: minimum maintainer/community surface with explicit project limits.
- Create `.github/workflows/ci.yml`: supported Python matrix, package build, focused tests, full tests, and documentation smoke checks.
- Create `.github/workflows/release.yml`: tag-triggered build and PyPI trusted publishing without embedding credentials.
- Create `.github/ISSUE_TEMPLATE/bug_report.yml`, `.github/ISSUE_TEMPLATE/feature_request.yml`, and `.github/PULL_REQUEST_TEMPLATE.md`: structured contribution intake.
- Create `tests/test_readme_contract.py`: enforce the canonical install command, MemoryAgent positioning, required links, and absence of the two colliding package names as install commands.

---

### Task 1: Establish collision-free package metadata and console installation

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/memory_agent/__init__.py` only if a version constant is needed by the metadata test
- Create: `tests/test_package_metadata.py`

- [ ] **Step 1: Write failing metadata tests**

Add tests that inspect `importlib.metadata.metadata("alfredo-memory-agent")` in an installed/editable environment and assert:

```python
assert metadata["Name"] == "alfredo-memory-agent"
assert "MemoryAgent" in metadata["Summary"]
assert metadata["Requires-Python"] == ">=3.11"
```

Also assert that the installed distribution exposes an `alfredo` console script targeting `memory_agent.cli.commands:cli`, while `import memory_agent` and its existing `__all__` exports still work.

- [ ] **Step 2: Run the focused test and observe the failure**

Run:

```bash
python -m pytest tests/test_package_metadata.py -q
```

Expected: FAIL because the current project distribution is named `memory-agent` and no `alfredo` console script is declared.

- [ ] **Step 3: Implement the public package contract**

In `pyproject.toml`:

```toml
[project]
name = "alfredo-memory-agent"
description = "An open-source MemoryAgent that learns, remembers, forgets, and explains every recall"
requires-python = ">=3.11"

[project.scripts]
alfredo = "memory_agent.cli.commands:cli"
```

Keep the Python import as `memory_agent`. Add repository, documentation, issues, and changelog URLs only when each target exists. Keep `numpy`, `click`, and `rich` as base dependencies; move `sentence-transformers` to the `semantic` extra so the first offline install does not download a transformer stack. Make `all` include `semantic`, `mcp`, `llm`, and `cloud` dependencies without duplicating contradictory declarations.

Add classifiers for Python 3.11–3.14 and a `Development Status :: 4 - Beta` classifier only if it matches the actual release state. Do not rename the import package or remove existing extras.

- [ ] **Step 4: Build and rerun focused tests**

Run:

```bash
python -m pip install -e .
python -m pytest tests/test_package_metadata.py -q
python -m build
python -m twine check dist/*
```

Expected: the editable package reports `alfredo-memory-agent`, the console entry point resolves, and the wheel/sdist pass metadata validation.

- [ ] **Step 5: Commit the package contract**

```bash
git add pyproject.toml src/memory_agent/__init__.py tests/test_package_metadata.py
git commit -m "feat: publish Alfredo MemoryAgent package metadata"
```

### Task 2: Make the branded offline first run and demo stable

**Files:**
- Modify: `src/memory_agent/cli/commands.py`
- Modify: `examples/demo_basic.py`
- Modify: `examples/demo_hackathon.py`
- Create: `examples/demo_lifecycle.py`
- Modify: `tests/test_cli_quickstart.py`
- Modify: `tests/test_documentation_commands.py`

- [ ] **Step 1: Add failing branded CLI and lifecycle tests**

Extend the quickstart tests to invoke the Click command through both `cli` and the installed `alfredo` entry point where available. Assert output contains `Alfredo MemoryAgent`, `Remembered:`, and the deterministic provider label, and that the temporary SQLite database is closed and cleaned up.

Add a lifecycle test for `examples/demo_lifecycle.py` that executes it in a subprocess with a temporary home and asserts stable markers for stored preference, cross-session recall, superseded/archived memory, selected/dropped context IDs, and clean exit.

- [ ] **Step 2: Run focused tests and capture failures**

```bash
python -m pytest tests/test_cli_quickstart.py tests/test_documentation_commands.py -q
python examples/demo_lifecycle.py
```

Expected: the current output is branded only as `MemoryAgent`, and the new lifecycle script is absent.

- [ ] **Step 3: Implement the minimal branded path**

Update CLI help and quickstart output to use the canonical wording:

```text
Alfredo MemoryAgent — offline quickstart
Persistent memory • timely forgetting • bounded, explainable recall
```

Keep `--offline` explicit and preserve `python -m memory_agent`. Implement `examples/demo_lifecycle.py` with `MemoryAgent` and the deterministic embedding configuration, four sessions, and explicit printed sections:

```text
[1] learn preference
[2] recall across session
[3] supersede stale preference
[4] bounded context and trust evidence
```

The script must delete its temporary database in `finally`, never require an API key, and avoid nondeterministic timestamps in displayed output. Update existing demos only where their output claims or cleanup violate this contract.

- [ ] **Step 4: Run focused tests and demos again**

```bash
python -m pytest tests/test_cli_quickstart.py tests/test_documentation_commands.py tests/test_video_demo.py -q
python examples/demo_lifecycle.py
```

Expected: all focused tests pass and the demo prints the four lifecycle sections without external services.

- [ ] **Step 5: Commit the first-run experience**

```bash
git add src/memory_agent/cli/commands.py examples/demo_basic.py examples/demo_hackathon.py examples/demo_lifecycle.py tests/test_cli_quickstart.py tests/test_documentation_commands.py
git commit -m "feat: add branded offline MemoryAgent lifecycle demo"
```

### Task 3: Rebuild the README around the MemoryAgent conversion funnel

**Files:**
- Modify: `README.md`
- Create: `docs/assets/alfredo-memory-lifecycle.svg`
- Create: `tests/test_readme_contract.py`

- [ ] **Step 1: Add README contract tests**

Assert that `README.md` contains the canonical brand statement, the exact install command, the offline quickstart command, links to the lifecycle demo, MCP integration, benchmark, license, security model, and contribution guide. Assert that `pip install alfredo` and `pip install memory-agent` do not appear as the primary installation command; the only canonical public distribution command is `pip install alfredo-memory-agent`.

- [ ] **Step 2: Run the README contract test**

```bash
python -m pytest tests/test_readme_contract.py -q
```

Expected: FAIL until the README is rewritten around the new canonical contract.

- [ ] **Step 3: Replace the opening with the visual hero and exact positioning**

Use this copy near the top:

```markdown
# Alfredo

### The open-source MemoryAgent for agents that learns, remembers, forgets, and explains why.

Alfredo gives AI agents persistent, selective, and evolving memory across
multi-turn and cross-session interactions. It turns conversations into
structured experience, retrieves only what matters, supersedes contradictions,
forgets stale information, and packs critical memories into bounded context.

```bash
pip install alfredo-memory-agent
alfredo --offline quickstart
```

[30-second lifecycle demo](./examples/demo_lifecycle.py) · [MCP integration](./INTEGRATION.md) · [Benchmark](./benchmarks/alfredos_vault/)
```

Add verified badges only for existing CI, PyPI, Python, license, and release targets. Embed `docs/assets/alfredo-memory-lifecycle.svg` immediately after the hero.

- [ ] **Step 4: Reorganize the rest of README into user paths**

Use this order:

1. What Alfredo is and is not;
2. two-command offline quickstart;
3. the lifecycle demo;
4. why it is different from history-only and basic RAG;
5. how retrieval, trust, forgetting, supersession, and context budgets work;
6. benchmark evidence and its synthetic-data limitation;
7. Python SDK and MCP recipes;
8. architecture/module map;
9. testing and development;
10. security/privacy boundaries;
11. contribution, roadmap, license, and future hosted work.

Keep the existing verified numbers (25 users, 5,000 memories, 500 questions, 198 tests only if the current run confirms it) and never claim 100k stars, production privacy, or universal accuracy. Replace hackathon-first language with MemoryAgent-first language while preserving the submission links.

- [ ] **Step 5: Run documentation tests and inspect rendered Markdown**

```bash
python -m pytest tests/test_readme_contract.py tests/test_documentation_commands.py -q
```

Expected: PASS, with no broken local links or stale package names.

- [ ] **Step 6: Commit the README conversion funnel**

```bash
git add README.md docs/assets/alfredo-memory-lifecycle.svg tests/test_readme_contract.py
git commit -m "docs: redesign README around Alfredo MemoryAgent"
```

### Task 4: Align integration, architecture, and community documentation

**Files:**
- Modify: `INTEGRATION.md`
- Modify: `docs/ARCHITECTURE.md`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `CHANGELOG.md`
- Create: `ROADMAP.md`

- [ ] **Step 1: Add the canonical installation and API paths**

Update integration docs to begin with:

```bash
python -m pip install alfredo-memory-agent
alfredo --offline quickstart
```

Keep `python -m memory_agent` as a compatibility path. Add copyable MCP stdio and HTTP recipes for Hermes, Claude Desktop, Cursor, and generic MCP clients. Every recipe must state whether it needs the `mcp` extra and must show namespace behavior where applicable.

- [ ] **Step 2: Document the real MemoryAgent lifecycle**

Update architecture docs to show:

```text
perceive → extract → validate/trust → store → retrieve → pack context
        → reinforce useful memories → supersede contradictions
        → decay/archive stale memories
```

Document SQLite local-first storage, provider/dimension guards, deterministic offline embeddings, selected/dropped IDs, and the boundary between synthetic benchmark evidence and production privacy controls.

- [ ] **Step 3: Add maintainer and contributor policies**

`CONTRIBUTING.md` must include editable install, focused test commands, full test command, README command smoke checks, commit expectations, and how to add a provider without duplicating lifecycle logic. `SECURITY.md` must explain reporting, namespaces, explicit forget, sensitive data limitations, and the prohibition against treating the synthetic benchmark as a security audit. `CODE_OF_CONDUCT.md` must use a complete standard enforcement policy. `CHANGELOG.md` must record the current release line without inventing unreleased features. `ROADMAP.md` must separate the current local SDK from future dashboard/hosting/TypeScript work.

- [ ] **Step 4: Run documentation smoke checks**

```bash
python -m pytest tests/test_documentation_commands.py tests/test_readme_contract.py -q
```

Expected: PASS and no documented command references the rejected PyPI names.

- [ ] **Step 5: Commit the public documentation surface**

```bash
git add INTEGRATION.md docs/ARCHITECTURE.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md CHANGELOG.md ROADMAP.md
git commit -m "docs: align Alfredo integration and contributor experience"
```

### Task 5: Add CI, release automation, and GitHub contribution surfaces

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

- [ ] **Step 1: Define CI checks before implementation**

The CI workflow must run on pull requests and pushes to `main`, use Python 3.11, 3.12, 3.13, and 3.14 where available, install the package with the minimal dependency set, run offline/documentation tests, run the complete suite on one matrix job, build the wheel and sdist, and run `twine check`. It must not require API keys or download a transformer model.

- [ ] **Step 2: Define release automation without secret-in-repository credentials**

The release workflow must trigger on tags matching `v*`, run the same package build and metadata checks, and publish through PyPI trusted publishing/OIDC. It must not store a PyPI token in YAML or commit generated `dist/` artifacts. The workflow must fail before publishing if the distribution metadata is not `alfredo-memory-agent`.

- [ ] **Step 3: Add structured issue and pull request templates**

Bug reports must ask for Alfredo version, Python version, OS, provider, namespace, reproduction, and expected/actual lifecycle behavior. Feature requests must ask which MemoryAgent contract they improve and whether they affect storage, retrieval, trust, forgetting, context budgets, or integrations. Pull requests must require tests, docs updates, security/privacy impact, and confirmation that no provider-specific lifecycle duplicate was added.

- [ ] **Step 4: Validate workflow syntax and local equivalents**

Run the local commands represented by CI:

```bash
python -m pip install -e .
python -m pytest tests/test_cli_quickstart.py tests/test_documentation_commands.py tests/test_readme_contract.py -q
python -m pytest tests/ -q
python -m build
python -m twine check dist/*
```

Validate YAML parsing with the repository’s available YAML tooling; if none is installed, use Python’s standard file checks and inspect the workflow structure manually without introducing a runtime dependency solely for linting.

- [ ] **Step 5: Commit GitHub automation**

```bash
git add .github
git commit -m "ci: add Alfredo test and PyPI release workflows"
```

### Task 6: Verify clean-install adoption and release evidence

**Files:**
- Modify: `README.md` only if clean-install output reveals a stale command
- Modify: `tests/test_submission_artifacts.py` only if release links require a regression assertion
- No new runtime code unless a preceding focused test identifies a real defect

- [ ] **Step 1: Build a clean distribution**

```powershell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
```

On Windows, use an equivalent PowerShell cleanup if `cmd` syntax is unavailable. Confirm the wheel contains `memory_agent`, the metadata name is `alfredo-memory-agent`, and the console script is present.

- [ ] **Step 2: Test the wheel outside the checkout**

Create a fresh virtual environment in a temporary directory, install the locally built wheel, and run:

```bash
python -m pip install dist/alfredo_memory_agent-*.whl
alfredo --offline quickstart
python -m memory_agent --offline quickstart
```

Expected: both commands complete without API keys, transformer downloads, or imports from the source checkout.

- [ ] **Step 3: Run the reproducible benchmark path**

```bash
alfredo --offline benchmark compare --users benchmarks/alfredos_vault/users.json --memories benchmarks/alfredos_vault/memories.jsonl --questions benchmarks/alfredos_vault/evaluation_questions.jsonl --report .alfredo/release-comparison.json --seed 42 --run release-local
```

Confirm all three strategies appear, dataset/config hashes are present, selected/dropped IDs and trust evidence are serialized, and no report contains real user data.

- [ ] **Step 4: Run the complete verification suite**

```bash
python -m pytest tests/ -q
```

Expected: all existing and new tests pass. Remove temporary virtual environments, reports, build directories, and pytest artifacts before status inspection.

- [ ] **Step 5: Review the final public contract and commit verification metadata only when required**

Check that `git status --short` contains no generated artifacts, README commands match the wheel-tested commands, package names are consistent, and the release workflow is tag-ready. Do not commit benchmark output or credentials.

---

## Plan self-review

- **Spec coverage:** README hero/visual/demo is covered by Task 3; one-line install and PyPI metadata by Tasks 1 and 6; offline quickstart and lifecycle by Task 2; benchmark proof and limitations by Tasks 3 and 6; integrations/security/community by Task 4; CI/release by Task 5; full validation by Task 6. Dashboard, SaaS, billing, hosted storage, and TypeScript remain explicitly out of scope.
- **Collision handling:** the plan uses `alfredo-memory-agent`, because `alfredo` and `memory-agent` are already occupied on PyPI by unrelated projects. The brand remains Alfredo and the import remains `memory_agent`.
- **No placeholders:** every task names exact files, commands, expected outcomes, and commit boundaries. No generated asset, badge, or external service is assumed to exist without a validation step.
- **Type/API consistency:** the console script targets the existing `memory_agent.cli.commands:cli`; runtime imports remain `memory_agent`; the README and integration docs use the same commands; offline mode remains explicit.
- **Risk control:** moving `sentence-transformers` to an optional extra is included only with focused installation/quickstart tests, preventing the visual adoption work from silently breaking semantic provider behavior.
