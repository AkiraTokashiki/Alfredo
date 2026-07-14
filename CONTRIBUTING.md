# Contributing to Alfredo MemoryAgent

Thank you for improving Alfredo. This repository is a local-first Python SDK; contributions should preserve the `alfredo-memory-agent` distribution name, the `memory_agent` import namespace, and the explainable lifecycle documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Local setup

Use Python 3.11 or newer. From a fresh checkout, an editable install with quoted extras is safe in Windows PowerShell, Windows CMD, and POSIX shells:

```powershell
python -m pip install -e ".[semantic,mcp]"
```

The `semantic` extra installs the optional sentence-transformers provider and `mcp` installs the MCP server dependencies. For the deterministic core only, use:

```powershell
python -m pip install -e .
```

Do not document or test the historical bare distribution names `alfredo` or `memory-agent` as install targets. The release install contract is `python -m pip install alfredo-memory-agent`; PyPI publication status is not implied by a checkout.

## Focused checks first

Run only the relevant focused tests while iterating. Documentation-only changes should begin with:

```powershell
python -m pytest tests/test_public_docs_contract.py tests/test_readme_contract.py tests/test_documentation_commands.py -q --basetemp=.pytest-contrib
```

For code changes, add or update a narrowly scoped test and run its path (for example, `python -m pytest tests/test_retrieval.py -q`). Keep temporary output outside tracked files. Before requesting review, run the full test suite:

```powershell
python -m pytest
```

Also perform a README/docs smoke check: run `python -m memory_agent --offline quickstart`, inspect copyable commands for rejected installs, and check that every relative Markdown link points to a file or directory in the repository. Offline checks must not download model weights or require API keys.

## Pull requests and commits

Describe the user-visible behavior, the local/offline behavior, and any migration or compatibility impact. Link the focused tests and explain any intentionally untested integration (for example, a client-specific MCP UI). Keep commits small and imperative, with a subject that says what changed; do not mix formatting churn or generated benchmark reports into a behavioral change. A pull request should be reviewable from its diff and should state the exact commands used for verification.

## Adding a provider or adapter

1. Start with the existing ports in `src/memory_agent/ports.py` and the configuration/provider factory. Define the provider's identity and vector dimension explicitly.
2. Add focused tests for encoding, provider/dimension mismatch guards, failure behavior, and any required optional dependency. Keep offline deterministic behavior working without the provider.
3. Inject the provider into `MemoryAgent`; do not add a second orchestrator, extraction pass, trust policy, context packer, reinforcement loop, supersession rule, or decay cycle. Providers must participate in the existing lifecycle **without duplicating the lifecycle**.
4. Document installation only with the actual optional extra and preserve the `memory_agent` module compatibility path. Do not claim a hosted service or PyPI availability that has not been released.

Namespace handling, selected/dropped IDs, evidence, explicit forget, and synthetic benchmark boundaries are public behavior. Changes to those contracts need tests and corresponding documentation updates.
