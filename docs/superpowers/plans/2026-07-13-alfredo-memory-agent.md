# Alfredo Memory Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Alfredo’s existing hackathon memory prototype into a stable open-source SDK with explainable lifecycle behavior, reproducible benchmark comparisons, and a five-minute local quickstart.

**Architecture:** Preserve the existing orchestrator, SQLite store, embeddings, retrieval, consolidation, forgetting, CLI, MCP, and LLM adapters, but introduce small public protocols and explicit result metadata at the boundary. SQLite and local embeddings remain the defaults; adapters depend on the domain facade rather than reimplementing lifecycle logic. The first release does not build a SaaS dashboard or managed multi-tenant service.

**Tech Stack:** Python 3.11+, dataclasses/protocols, SQLite WAL, NumPy, sentence-transformers, Click, Rich, MCP, pytest, pytest-cov, JSON/JSONL benchmark artifacts.

---

## File map before implementation

- Modify `src/memory_agent/models.py`: add stable memory metadata and explainable retrieval decision types while retaining backward-compatible dataclasses.
- Create `src/memory_agent/ports.py`: runtime-checkable protocols for storage, embeddings, retrieval, and trust decisions.
- Modify `src/memory_agent/agent/orchestrator.py`: dependency injection, namespace isolation, explicit session parameters, and structured lifecycle results.
- Modify `src/memory_agent/core/memory_store.py`: schema migration, namespace filtering, supersession/forget metadata, and transactional helpers.
- Modify `src/memory_agent/core/retrieval.py`: namespace-aware retrieval and per-result scoring signals.
- Modify `src/memory_agent/core/context_budget.py`: expose packing decisions and exact budget accounting.
- Modify `src/memory_agent/integrations/mcp_server.py`: route all tools through the public agent facade and expose explainability fields.
- Modify `src/memory_agent/cli/commands.py`: first-run flow, deterministic mode selection, namespace option, and inspection commands.
- Modify `src/memory_agent/core/config.py` and `pyproject.toml`: explicit defaults, optional dependency groups, and deterministic embedding fallback configuration.
- Create `src/memory_agent/core/deterministic_embeddings.py`: lightweight offline embedding provider for demos/tests without model download.
- Modify `src/memory_agent/benchmark.py`: baseline runners, versioned metrics, latency/token accounting, and comparison report.
- Create `benchmarks/alfredos_vault/baselines/`: reproducible baseline implementations and result metadata.
- Create or modify tests under `tests/`: contracts, isolation, lifecycle explainability, context budgets, benchmark comparisons, CLI/MCP smoke coverage.
- Modify `README.md`, `INTEGRATION.md`, `docs/ARCHITECTURE.md`, and `ABOUT_THE_PROJECT.md`: five-minute quickstart, public API, security/privacy model, benchmark reproduction, and extension guide.
- Modify `requirements.txt` only if the project’s editable install path requires a pinned test/runtime dependency; do not duplicate `pyproject.toml` dependency declarations unnecessarily.

---

### Task 1: Establish public contracts and result types

**Files:**
- Create: `src/memory_agent/ports.py`
- Modify: `src/memory_agent/models.py`
- Modify: `src/memory_agent/__init__.py`
- Test: `tests/test_public_contracts.py`

- [ ] **Step 1: Write failing contract tests**

Add tests asserting that `MemoryStorePort`, `EmbeddingPort`, `RetrievalPort`, and `TrustPolicyPort` are importable protocols, and that `MemoryRecord`, `SearchResult`, and the new explainability type can serialize to JSON-safe dictionaries without leaking private implementation objects.

- [ ] **Step 2: Run the focused test**

Run: `python -m pytest tests/test_public_contracts.py -v`
Expected: FAIL because the protocols and explainability type do not exist.

- [ ] **Step 3: Implement minimal protocols and models**

Define protocols with only methods used by the orchestrator: store/read/update memory, encode text, retrieve candidates, and evaluate trust. Add an immutable `RetrievalEvidence` dataclass containing `score`, `semantic_score`, `recency_score`, `importance_score`, `strength_score`, `matched_by`, `trust`, and `reason`. Add optional `namespace`, `confidence`, `sensitivity`, `source`, `superseded_by`, and `last_decision_reason` fields with defaults so current callers continue to construct records.

- [ ] **Step 4: Re-run the focused test**

Run: `python -m pytest tests/test_public_contracts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/memory_agent/ports.py src/memory_agent/models.py src/memory_agent/__init__.py tests/test_public_contracts.py && git commit -m "feat: define public memory contracts"`

### Task 2: Add schema migration and namespace isolation

**Files:**
- Modify: `src/memory_agent/core/memory_store.py`
- Modify: `src/memory_agent/models.py`
- Test: `tests/test_memory_store.py`, `tests/test_namespace_isolation.py`

- [ ] **Step 1: Write failing isolation and migration tests**

Test that two users can store identical content in the same SQLite file and each retrieval only returns the caller’s namespace. Test initialization of an existing database preserves old memories and adds new columns through an idempotent migration.

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_memory_store.py tests/test_namespace_isolation.py -v`
Expected: FAIL because the schema has no namespace/migration behavior and store methods do not filter it.

- [ ] **Step 3: Implement a versioned SQLite migration**

Add a schema version table or equivalent idempotent migration routine. Add `namespace`, `confidence`, `sensitivity`, `source`, `superseded_by`, and `last_decision_reason` columns with safe defaults. Update all active-memory reads, counts, session links, embedding lookup, archive, forget, and update methods to accept a namespace and include it in predicates. Keep WAL mode and transaction semantics.

- [ ] **Step 4: Re-run store tests**

Run: `python -m pytest tests/test_memory_store.py tests/test_namespace_isolation.py -v`
Expected: PASS, including all pre-existing store tests.

- [ ] **Step 5: Commit**

Run: `git add src/memory_agent/core/memory_store.py src/memory_agent/models.py tests/test_memory_store.py tests/test_namespace_isolation.py && git commit -m "feat: isolate memory namespaces and migrate schema"`

### Task 3: Make retrieval and context packing explainable

**Files:**
- Modify: `src/memory_agent/core/retrieval.py`
- Modify: `src/memory_agent/core/context_budget.py`
- Modify: `src/memory_agent/models.py`
- Test: `tests/test_retrieval.py`, `tests/test_context_budget.py`, `tests/test_retrieval_explainability.py`

- [ ] **Step 1: Write failing evidence tests**

Assert that each selected result exposes its component scores, matched signals, trust decision, and reason. Assert that context packing reports selected IDs, dropped IDs, used characters, reserved characters, and the configured limit. Assert namespace filtering is applied before ranking.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_retrieval.py tests/test_context_budget.py tests/test_retrieval_explainability.py -v`
Expected: FAIL on missing evidence and packing metadata.

- [ ] **Step 3: Implement evidence propagation**

Preserve the existing weighted score and MMR ordering. Store the semantic, recency, importance, and strength components on each result; populate `matched_by` using thresholds tied to the configured minimum score. Apply trust filtering before context packing. Extend `RecallPacket` with deterministic accounting fields and keep existing `selected`/text properties intact.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_retrieval.py tests/test_context_budget.py tests/test_retrieval_explainability.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/memory_agent/core/retrieval.py src/memory_agent/core/context_budget.py src/memory_agent/models.py tests/test_retrieval.py tests/test_context_budget.py tests/test_retrieval_explainability.py && git commit -m "feat: expose explainable retrieval evidence"`

### Task 4: Inject dependencies through the orchestrator

**Files:**
- Modify: `src/memory_agent/agent/orchestrator.py`
- Modify: `src/memory_agent/core/config.py`
- Modify: `src/memory_agent/__init__.py`
- Test: `tests/test_agent.py`, `tests/test_agent_dependencies.py`

- [ ] **Step 1: Write failing facade tests**

Construct `MemoryAgent` with a fake store and fake embedder implementing the new protocols. Assert `perceive` accepts `namespace`/`user_id`, stores and retrieves only that namespace, returns structured evidence, and still supports the existing `MemoryAgent(db_path=...)` constructor.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_agent.py tests/test_agent_dependencies.py -v`
Expected: FAIL because the constructor always creates concrete dependencies and the result has no namespace/evidence fields.

- [ ] **Step 3: Implement dependency injection without duplicating lifecycle**

Add optional `store`, `embedder`, `retrieval`, and `trust_policy` constructor parameters. Use concrete defaults only when omitted. Thread `namespace` through extraction, consolidation, retrieval, decay, session links, and counts. Return the existing dictionary keys plus structured evidence and lifecycle decisions. Do not remove existing methods used by CLI or MCP.

- [ ] **Step 4: Run focused regression tests**

Run: `python -m pytest tests/test_agent.py tests/test_agent_dependencies.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/memory_agent/agent/orchestrator.py src/memory_agent/core/config.py src/memory_agent/__init__.py tests/test_agent.py tests/test_agent_dependencies.py && git commit -m "feat: inject memory agent dependencies"`

### Task 5: Add offline deterministic first-run mode

**Files:**
- Create: `src/memory_agent/core/deterministic_embeddings.py`
- Modify: `src/memory_agent/core/embeddings.py`
- Modify: `src/memory_agent/core/config.py`
- Modify: `src/memory_agent/cli/commands.py`
- Modify: `pyproject.toml`
- Test: `tests/test_deterministic_embeddings.py`, `tests/test_cli_quickstart.py`

- [ ] **Step 1: Write failing offline-mode tests**

Assert deterministic equal inputs produce equal vectors, different inputs produce bounded vectors, and a CLI demo can store and recall a memory without downloading a transformer model or requiring an API key.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_deterministic_embeddings.py tests/test_cli_quickstart.py -v`
Expected: FAIL because no offline provider/CLI switch exists.

- [ ] **Step 3: Implement the provider and explicit switch**

Implement a deterministic hashed-token vector with the configured dimension and cosine-compatible normalization. Add `--offline` or an equivalent config setting that selects it explicitly; do not silently replace a configured production model. Make the quickstart command use a temporary SQLite vault and print the recalled memory.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_deterministic_embeddings.py tests/test_cli_quickstart.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/memory_agent/core/deterministic_embeddings.py src/memory_agent/core/embeddings.py src/memory_agent/core/config.py src/memory_agent/cli/commands.py pyproject.toml tests/test_deterministic_embeddings.py tests/test_cli_quickstart.py && git commit -m "feat: add offline first-run mode"`

### Task 6: Harden MCP and CLI adapters around the public facade

**Files:**
- Modify: `src/memory_agent/integrations/mcp_server.py`
- Modify: `src/memory_agent/cli/commands.py`
- Modify: `INTEGRATION.md`
- Test: `tests/test_mcp_server.py`, `tests/test_cli_commands.py`

- [ ] **Step 1: Write failing adapter tests**

Assert MCP store/search/perceive tools accept a namespace and return JSON-safe evidence. Assert CLI `stats`, `search`, and `forget` show namespace and trust/lifecycle reasons. Assert adapters do not instantiate a second retrieval or forgetting implementation.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_mcp_server.py tests/test_cli_commands.py -v`
Expected: FAIL on new parameters and output fields.

- [ ] **Step 3: Route adapters through `MemoryAgent`**

Add optional namespace arguments, serialize dataclasses through their public `to_dict` methods, and expose selected/dropped memory IDs and trust reasons. Preserve existing tool names and CLI commands. Document stdio and HTTP examples with a namespace.

- [ ] **Step 4: Run adapter tests**

Run: `python -m pytest tests/test_mcp_server.py tests/test_cli_commands.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/memory_agent/integrations/mcp_server.py src/memory_agent/cli/commands.py INTEGRATION.md tests/test_mcp_server.py tests/test_cli_commands.py && git commit -m "feat: expose explainable memory adapters"`

### Task 7: Add reproducible benchmark baselines and metrics

**Files:**
- Modify: `src/memory_agent/benchmark.py`
- Create: `benchmarks/alfredos_vault/baselines/raw_history.py`
- Create: `benchmarks/alfredos_vault/baselines/semantic_rag.py`
- Create: `benchmarks/alfredos_vault/baselines/__init__.py`
- Modify: `benchmarks/alfredos_vault/README.md` or create it if absent
- Test: `tests/test_benchmark.py`, `tests/test_benchmark_baselines.py`

- [ ] **Step 1: Write failing benchmark tests**

Assert the benchmark runner can execute raw-history, semantic-RAG, and Alfredo strategies against the existing synthetic questions, emits a versioned JSON report, records pass/fail behavior, token/character context size, latency samples, and security events, and produces deterministic answers when deterministic embeddings are selected.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_benchmark.py tests/test_benchmark_baselines.py -v`
Expected: FAIL because only the current Alfredo evaluator exists.

- [ ] **Step 3: Implement baseline runners and report schema**

Keep benchmark fixtures synthetic. Define one strategy protocol taking a user, query, and memory set and returning answer, retrieved IDs, ignored IDs, confidence, context size, and behavior. Add raw-history and semantic-only implementations. Record package version, benchmark version, dataset hashes, configuration, p50/p95 latency, context size, and aggregate metrics. Preserve the existing seed/run CLI commands and add a comparison command.

- [ ] **Step 4: Run benchmark tests**

Run: `python -m pytest tests/test_benchmark.py tests/test_benchmark_baselines.py -v`
Expected: PASS.

- [ ] **Step 5: Run the reproducible sample benchmark**

Run: `python -m memory_agent --db .alfredo/test-vault.db benchmark compare --users benchmarks/alfredos_vault/users.json --memories benchmarks/alfredos_vault/memories.jsonl --questions benchmarks/alfredos_vault/evaluation_questions.jsonl --report .alfredo/test-comparison.json --offline`
Expected: JSON report with all three strategies, aggregate quality/security/context/latency fields, and no external API requirement.

- [ ] **Step 6: Commit**

Run: `git add src/memory_agent/benchmark.py benchmarks/alfredos_vault/baselines benchmarks/alfredos_vault/README.md tests/test_benchmark.py tests/test_benchmark_baselines.py && git commit -m "feat: add reproducible memory baselines"`

### Task 8: Document the five-minute developer path and security model

**Files:**
- Modify: `README.md`
- Modify: `INTEGRATION.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `ABOUT_THE_PROJECT.md`
- Test: `tests/test_documentation_commands.py`

- [ ] **Step 1: Write documentation smoke tests**

Add tests that extract fenced shell commands from the quickstart and verify the primary offline command references existing files/options. Add a test that the documented benchmark comparison command includes all required strategies and a report path.

- [ ] **Step 2: Run documentation tests**

Run: `python -m pytest tests/test_documentation_commands.py -v`
Expected: FAIL until the documented commands and parser agree.

- [ ] **Step 3: Rewrite the developer path**

Place a short offline quickstart at the top of `README.md`: install editable package, run one command, observe a cross-turn recall, inspect evidence, then opt into MCP/LLM. Add API examples with namespace isolation, provider extension guidance, privacy/security behavior, benchmark comparison, expected output, and troubleshooting for model downloads. Update architecture docs to show protocols, trust filtering, evidence, and baseline evaluation.

- [ ] **Step 4: Run documentation tests and command smoke test**

Run: `python -m pytest tests/test_documentation_commands.py -v`
Run: `python -m memory_agent --offline demo`
Expected: PASS and a visible stored/recalled memory with evidence fields.

- [ ] **Step 5: Commit**

Run: `git add README.md INTEGRATION.md docs/ARCHITECTURE.md ABOUT_THE_PROJECT.md tests/test_documentation_commands.py && git commit -m "docs: document Alfredo developer quickstart"`

### Task 9: Run targeted verification and prepare the release candidate

**Files:**
- Modify: `pyproject.toml` only if metadata/version/test configuration needs correction.
- Modify: `HISTORY.md` with the verified release notes.
- Test: all changed tests plus package smoke commands.

- [ ] **Step 1: Run the targeted suite**

Run: `python -m pytest tests/test_public_contracts.py tests/test_memory_store.py tests/test_namespace_isolation.py tests/test_retrieval.py tests/test_context_budget.py tests/test_retrieval_explainability.py tests/test_agent.py tests/test_agent_dependencies.py tests/test_deterministic_embeddings.py tests/test_cli_quickstart.py tests/test_mcp_server.py tests/test_cli_commands.py tests/test_benchmark.py tests/test_benchmark_baselines.py tests/test_documentation_commands.py -v`
Expected: PASS with no skipped contract tests.

- [ ] **Step 2: Run the offline end-to-end scenario**

Run: `python -m memory_agent --offline demo`
Expected: one memory is stored, a later turn retrieves only the correct namespace, the output includes evidence, and no API key/model download is required.

- [ ] **Step 3: Run the benchmark comparison smoke test**

Run the Task 7 comparison command against the checked-in synthetic fixtures.
Expected: report is generated and includes raw-history, semantic-RAG, and Alfredo results.

- [ ] **Step 4: Update verified history**

Add only observed behavior, commands, report schema, and compatibility notes to `HISTORY.md`; do not claim adoption or star counts.

- [ ] **Step 5: Commit the release-candidate verification**

Run: `git add pyproject.toml HISTORY.md && git commit -m "chore: verify Alfredo SDK release candidate"`

---

## Plan self-review

- **Spec coverage:** SDK facade and contracts are Tasks 1 and 4; namespace and lifecycle metadata are Task 2; explainability and bounded context are Task 3; adapters are Task 6; first-run experience is Task 5 and Task 8; benchmark baselines and metrics are Task 7; security/privacy documentation is Task 8; verification is Task 9. SaaS/dashboard work is explicitly excluded.
- **No placeholders:** every task names exact files, failing tests, commands, expected outcomes, implementation shape, and commit boundaries. No `TODO`, `TBD`, or “implement later” step is required.
- **Type consistency:** `RetrievalEvidence` is introduced before retrieval/orchestrator/adapters consume it; namespace fields are added before store and facade filtering; deterministic embeddings are injected through the same embedding protocol used by the concrete engine; benchmark strategy output is defined before baseline implementations.
- **Scope control:** this remains one release-candidate plan for the approved first-stage SDK + benchmark scope. Managed platform work is not mixed into the implementation tasks.
