# MemoryAgent Integration with LLMs and Hermes

## 1. MCP Server mode (recommended for Hermes)

MemoryAgent exposes a Model Context Protocol server with memory tools.
Any MCP client can use it, including Hermes, Claude Desktop, and Cursor.

### 1.1 Add it to Hermes

```bash
hermes mcp add memory-agent \
  --command python \
  --args "-m,memory_agent,mcp"
```

Or edit `~/.hermes/config.yaml` and add:

```yaml
mcp_servers:
  memory-agent:
    command: python
    args: ["-m", "memory_agent", "mcp"]
    # If Hermes runs in another environment:
    # command: "E:/CODE/MemoryAgent/.venv/Scripts/python.exe"
```

Then run `/reload-mcp` in the Hermes session. These tools become available:

- `memory__perceive` — process input and retrieve relevant memories
- `memory__search` — semantic memory search
- `memory__store` — explicitly store a memory
- `memory__stats` — show memory statistics
- `memory__forget` — delete/archive memory
- `memory__reinforce` — reinforce a memory


Todas las herramientas aceptan `namespace` opcional. El namespace se propaga por la
fachada `MemoryAgent`, por lo que las búsquedas, estadísticas y operaciones de ciclo
de vida nunca mezclan memorias de otros tenants. Por ejemplo, con stdio:

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory__search","arguments":{"query":"idioma preferido","namespace":"tenant-a"}}}
```

El mismo argumento funciona sin cambios en el transporte HTTP; el protocolo y los
nombres de herramientas son idénticos.
### 1.2 HTTP service mode (for remote access)

```bash
cd E:\CODE\MemoryAgent
.venv\Scripts\python -m memory_agent mcp --http --port 8090
# Listens on http://localhost:8090/mcp
```

Then configure Hermes:

```yaml
mcp_servers:
  memory-agent:
    url: http://localhost:8090/mcp
```

En HTTP, incluya `"namespace": "tenant-a"` en los argumentos JSON de
`memory__perceive`, `memory__search`, `memory__store`, `memory__stats`,
`memory__forget` o `memory__reinforce`. Las respuestas incluyen `namespace`,
`selected_ids`, `dropped_ids`, evidencia de confianza (`trust` y `reason`) y el
estado `lifecycle` cuando aplica.

### 1.3 Test the MCP server

```bash
# Verify stdio mode responds
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  .venv\Scripts\python -m memory_agent mcp

# Or from another terminal when HTTP mode is active
curl http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

---

## 2. Standalone LLM connector

The standalone connector runs a complete conversation with a real LLM while using MemoryAgent as persistent memory.

### 2.1 Configure an API key

```bash
# DeepSeek (default)
set DEEPSEEK_API_KEY=sk-...   # Windows CMD
export DEEPSEEK_API_KEY=sk-... # bash

# Or OpenRouter
set OPENROUTER_API_KEY=sk-...
```

### 2.2 Start an interactive session

```bash
cd E:\CODE\MemoryAgent
.venv\Scripts\python -m memory_agent.integrations.llm_connector
```

With an explicit provider:

```bash
# DeepSeek (default)
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  --provider deepseek

# OpenRouter
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  --provider openrouter --model openai/gpt-4o

# OpenAI
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  --provider openai --model gpt-4o

# Anthropic
set ANTHROPIC_API_KEY=sk-...
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  --provider anthropic --model claude-sonnet-4-20250514
```

### 2.3 Single query

```bash
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  -q "What do you know about me?"
```

### 2.4 Persistent database across sessions

By default, MemoryAgent stores runtime memory in the native memory vault.
Pass `--db` to use a specific database path:

```bash
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  --db "E:/CODE/MemoryAgent/my_memory.db"
```

---

## 3. Hermes skill integration

To make Hermes load MemoryAgent behavior automatically at startup:

### Create the skill

```bash
hermes skills create memory-agent-integration
```

Skill content:

```markdown
# MemoryAgent Integration

When the user asks about something you may have discussed before,
use the memory__search tool to find relevant memories.

When the user shares a preference or important fact,
use the memory__store tool to save it.

At the start of each session, use memory__stats to check
how many memories are stored.
```

### Load it in a session

```bash
hermes -s memory-agent-integration
```

---

## 4. Programmatic Python usage

```python
from memory_agent.agent.orchestrator import MemoryAgent

agent = MemoryAgent(db_path="my_memory.db")
agent.init_session("my-session")

# One turn: extract, retrieve, and store
result = agent.perceive("I like programming in Python")

print("Retrieved memories:", len(result["recollections"]))
for r in result["recollections"]:
    print(f"  [{r.memory.memory_type}] {r.memory.content}")

print(f"\nActive memories: {result['total_memories']}")
print(f"Turn: {result['turn_count']}")

agent.end_session()
agent.close()
```

---

## 5. Supported providers

| Provider | Environment variable | Default model |
|----------|----------------------|---------------|
| Qwen Cloud | `DASHSCOPE_API_KEY` | qwen-plus |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| OpenRouter | `OPENROUTER_API_KEY` | openai/gpt-4o |
| OpenAI | `OPENAI_API_KEY` | gpt-4o |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4 |

---

## 6. Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    Client                           │
│  (Hermes / Claude Desktop / Cursor / script.py)     │
└────────────────┬────────────────────────────────────┘
                 │ MCP protocol (stdio or HTTP)
                 ▼
┌─────────────────────────────────────────────────────┐
│              MemoryAgent MCP Server                  │
│                                                      │
│  Tools: perceive │ search │ store │ stats │ forget   │
└──────────────────┬──────────────────────────────────┘
                   │ internal calls
                   ▼
┌─────────────────────────────────────────────────────┐
│              MemoryAgent Core                        │
│                                                      │
│  Orchestrator → MemoryStore → SQLite                 │
│              → Embeddings → sentence-transformers    │
│              → Forgetting → Ebbinghaus curve         │
│              → Retrieval → Scoring + MMR             │
└─────────────────────────────────────────────────────┘
```

---

## 7. Production recommendations

1. Run the MCP server as a supervised service.
2. Back up the native SQLite vault.
3. Set a stable `ALFREDO_HOME` before deployment.
4. Use one database per agent/user boundary when memory isolation matters.
5. Monitor database size and retrieval latency as memories grow.
