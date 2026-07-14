# Alfredo MemoryAgent integration

This guide documents the entry points that exist in this repository. The distribution name is `alfredo-memory-agent`; the import and module namespace remains `memory_agent`. The examples use a local SQLite vault and do not imply a hosted Alfredo service.

## Install and run offline first

The release install contract starts with the canonical distribution name:

```bash
python -m pip install alfredo-memory-agent
alfredo --offline quickstart
```

The `alfredo` command is installed by the package. The module entry point remains compatible with existing scripts:

```bash
python -m memory_agent --offline quickstart
```

`--offline` selects deterministic hashed-token embeddings. The quickstart uses a temporary SQLite database unless `--db` is supplied, performs a cross-turn recall, and requires **no API key, network request, or model download**. No API keys are needed for any of the offline CLI, Python, or MCP examples below.

For a checkout, an editable install is useful during development. Quote extras so the command is safe in Windows PowerShell and POSIX shells:

```powershell
python -m pip install -e ".[mcp]"
```

The `mcp` extra is required only for the MCP server (`mcp` and `httpx`). It is not required for the offline CLI or the core Python API. The optional `semantic` extra enables `sentence-transformers`; it is not used by `--offline`:

```powershell
python -m pip install -e ".[semantic]"
alfredo --db .alfredo/memory.db chat
```

Keep databases separate when the embedding provider or dimension changes.

## Offline CLI

Global options come before the subcommand:

```bash
alfredo --offline --db .alfredo/memory.db stats --namespace tenant-a
alfredo --offline --db .alfredo/memory.db search "preferred language" --namespace tenant-a
alfredo --offline --db .alfredo/memory.db memories --namespace tenant-a
alfredo --offline --db .alfredo/memory.db forget 12 --namespace tenant-a
alfredo --offline --db .alfredo/memory.db benchmark compare \
  --users benchmarks/alfredos_vault/users.json \
  --memories benchmarks/alfredos_vault/memories.jsonl \
  --questions benchmarks/alfredos_vault/evaluation_questions.jsonl \
  --report .alfredo/benchmark-comparison.json --seed 42 --run local-offline
```

The module form is equivalent: replace `alfredo` with `python -m memory_agent`. `chat` is a local interactive session. `stats`, `search`, `memories`, and `forget` accept an optional namespace and operate through the same facade as the API. `benchmark compare` is a synthetic, deterministic comparison and must be run with `--offline` (or the global `--offline`).

## Python API

`MemoryAgent` is the public orchestration facade. It accepts `db_path`, a `MemoryAgentConfig`, and optional injected store, embedding, retrieval, and trust ports. A minimal local example is:

```python
from memory_agent.agent.orchestrator import MemoryAgent

agent = MemoryAgent(db_path=".alfredo/memory.db")
agent.init_session("assistant", namespace="tenant-a")

result = agent.perceive(
    "I prefer concise Python examples",
    namespace="tenant-a",
)
print(result["recollection_text"])
print(result["recall_packet"].selected_ids)

search = agent.search_memories("Python", namespace="tenant-a")
print(search["selected_ids"], search["dropped_ids"])
agent.end_session()
agent.close()
```

Useful facade methods are `init_session`, `end_session`, `perceive`, `store_memory`, `search_memories`, `list_memories`, `get_stats`, `reinforce_memory`, `forget_memory`, `explain_memory`, and `close`. `perceive` extracts and consolidates candidates, retrieves trusted memories, packs a bounded context, reinforces selected records, stores the interaction when appropriate, and applies decay/archive rules. Results expose lifecycle information, evidence, `selected_ids`, and `dropped_ids`.

`forget_memory(id, namespace=...)` archives the matching record in that namespace; it is an explicit lifecycle operation, not a promise to erase SQLite backups or application logs. See [SECURITY.md](SECURITY.md) for deletion and sensitive-data limits.

## MCP server prerequisites and behavior

MCP server mode requires the `mcp` extra. Install it from a release distribution with `python -m pip install "alfredo-memory-agent[mcp]"`, or from a checkout with the editable command above. The core offline CLI and Python API do **not** require this extra. The server itself uses deterministic offline embeddings only when configured with the global `--offline` option; otherwise it uses the configured embedding provider and may require its model dependency. MCP memory operations do not require an LLM API key.

The server exposes `memory__perceive`, `memory__search`, `memory__store`, `memory__stats`, `memory__forget`, and `memory__reinforce`. Every tool that accepts `namespace` passes it through the `MemoryAgent` facade. A namespace is a storage and session boundary: retrieval, statistics, store, forget, and reinforcement do not cross it. Responses include the effective namespace where applicable, plus lifecycle and evidence fields; perceive/search expose `selected_ids` and `dropped_ids`.

### Hermes recipe

Hermes supports both transports. The `mcp` extra is **required** for Alfredo's server; it is not needed by the offline CLI. Stdio is the simplest local, no-API-key path:

```bash
hermes mcp add memory-agent --command python --args "-m,memory_agent,mcp"
python -m memory_agent --offline mcp
```

For HTTP, start the server in one terminal and point Hermes at the reported `/mcp` endpoint:

```bash
python -m memory_agent --offline mcp --http --host localhost --port 8090
```

Use `namespace: "tenant-a"` in each tool call to keep Hermes sessions isolated. The HTTP transport is the server's SSE mode; it prints `http://localhost:8090/mcp` when it starts.

### Claude Desktop recipe

Claude Desktop can launch the same stdio command. The `mcp` extra is **required** for the server, while no extra and no API key are needed for the offline CLI itself. In Claude Desktop's MCP configuration, use an absolute interpreter when necessary:

```json
{
  "mcpServers": {
    "alfredo-memory": {
      "command": "python",
      "args": ["-m", "memory_agent", "mcp"],
      "env": {"MEMORY_AGENT_DB": ".alfredo/claude.db"}
    }
  }
}
```

The HTTP alternative is:

```bash
python -m memory_agent --offline mcp --http --host localhost --port 8090
```

Configure Claude Desktop's HTTP MCP URL as `http://localhost:8090/mcp` if your client supports HTTP MCP. Include `namespace` in every tool argument; an omitted namespace is a distinct `null` scope, not a wildcard.

### Cursor recipe

Cursor can use either a local stdio server or the HTTP endpoint. The `mcp` extra is **required** to run the server; offline operation needs no API key. A stdio entry is equivalent to:

```json
{
  "mcpServers": {
    "alfredo-memory": {
      "command": "python",
      "args": ["-m", "memory_agent", "mcp"],
      "env": {"MEMORY_AGENT_DB": ".alfredo/cursor.db"}
    }
  }
}
```

For HTTP, run:

```bash
python -m memory_agent --offline mcp --http --port 8090
```

Then set Cursor's MCP server URL to `http://localhost:8090/mcp` and pass a stable namespace such as `workspace-alfredo` on every `memory__search`, `memory__store`, `memory__perceive`, `memory__stats`, `memory__forget`, and `memory__reinforce` call.

### Generic MCP client recipe

Any MCP-compatible client can choose stdio or HTTP. The `mcp` extra is **required** for Alfredo's MCP server; it is not required for the core/offline CLI. Neither transport needs an API key for local deterministic operation.

Stdio configuration:

```json
{
  "command": "python",
  "args": ["-m", "memory_agent", "mcp"],
  "env": {"MEMORY_AGENT_DB": ".alfredo/generic.db"}
}
```

HTTP configuration:

```bash
python -m memory_agent --offline mcp --http --host localhost --port 8090
```

Connect to `http://localhost:8090/mcp`. In either transport, call for example:

```json
{
  "name": "memory__search",
  "arguments": {"query": "preferred language", "top_k": 5, "namespace": "tenant-a"}
}
```

The namespace is an explicit boundary, not a presentation label. Use the same value for all operations that belong to one agent or tenant, and use `memory__forget` for an explicit user request.

## Optional hosted LLM connector

The `llm` command is separate from offline memory and requires the provider's API key. It is not needed for MCP or the local SDK. Provider names accepted by the CLI are `qwencloud`, `deepseek`, `openrouter`, `openai`, and `anthropic`; configure the corresponding provider dependency and environment variable before using it. Keep API-key-bearing deployment configuration out of SQLite records and issue trackers.

## Verification

From a checkout, the deterministic lifecycle demo is available at [`examples/demo_lifecycle.py`](examples/demo_lifecycle.py). The synthetic comparison fixtures and their privacy boundary are described in [`README.md`](README.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). These links are repository paths, not hosted-service commitments.
