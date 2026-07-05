# Integracion de MemoryAgent con LLMs y Hermes

## 1. Como MCP Server (recomendado para Hermes)

MemoryAgent se expone como servidor MCP con herramientas de memoria.
Cualquier cliente MCP (Hermes, Claude Desktop, Cursor) puede usarlo.

### 1.1 Agregar a Hermes

```bash
hermes mcp add memory-agent \
  --command python \
  --args "-m,memory_agent,mcp"
```

O editar `~/.hermes/config.yaml` y agregar:

```yaml
mcp_servers:
  memory-agent:
    command: python
    args: ["-m", "memory_agent", "mcp"]
    # Si Hermes esta en otro entorno:
    # command: "E:/CODE/MemoryAgent/.venv/Scripts/python.exe"
```

Luego `/reload-mcp` en session Hermes y las herramientas aparecen:

- `memory__perceive` — procesa input + busca recuerdos
- `memory__search` — busqueda semantica
- `memory__store` — guardar explicitamente
- `memory__stats` — estadisticas
- `memory__forget` — eliminar memoria
- `memory__reinforce` — reforzar un recuerdo

### 1.2 Como servicio HTTP (para acceso remoto)

```bash
cd E:\CODE\MemoryAgent
.venv\Scripts\python -m memory_agent mcp --http --port 8090
# Escucha en http://localhost:8090/mcp
```

Luego en Hermes:

```yaml
mcp_servers:
  memory-agent:
    url: http://localhost:8090/mcp
```

### 1.3 Probar el MCP server

```bash
# Probar que responde
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  .venv\Scripts\python -m memory_agent mcp

# O desde otra terminal (HTTP mode activo):
curl http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

---

## 2. LLM Connector (standalone con DeepSeek/OpenAI)

Conversacion completa con LLM real usando MemoryAgent como memoria persistente.

### 2.1 Configurar API key

```bash
# DeepSeek (default, recomendado)
set DEEPSEEK_API_KEY=sk-...   # Windows CMD
export DEEPSEEK_API_KEY=sk-... # bash

# O OpenRouter
set OPENROUTER_API_KEY=sk-...
```

### 2.2 Iniciar sesion interactiva

```bash
cd E:\CODE\MemoryAgent
.venv\Scripts\python -m memory_agent.integrations.llm_connector
```

Con proveedor especifico:

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

### 2.3 Consulta unica

```bash
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  -q "Que sabes sobre mi?"
```

### 2.4 DB persistente entre sesiones

Por default la DB se guarda en el directorio actual (`memory_agent.db`).
Para usar una DB especifica:

```bash
.venv\Scripts\python -m memory_agent.integrations.llm_connector \
  --db "E:/CODE/MemoryAgent/mi_memoria.db"
```

---

## 3. Integracion como skill de Hermes

Para que Hermes cargue MemoryAgent automaticamente al iniciar:

### Crear el skill

```bash
hermes skills create memory-agent-integration
```

Contenido del skill:

```markdown
# MemoryAgent Integration

When the user asks about something you may have discussed before,
use the memory__search tool to find relevant memories.

When the user shares a preference or important fact,
use the memory__store tool to save it.

At the start of each session, use memory__stats to check
how many memories are stored.
```

### Cargar en session

```bash
hermes -s memory-agent-integration
```

---

## 4. Uso programatico (Python)

```python
from memory_agent.agent.orchestrator import MemoryAgent

agent = MemoryAgent(db_path="mi_memoria.db")
agent.init_session("mi-sesion")

# Un turno: extrae + busca + almacena
result = agent.perceive("Me gusta programar en Python")

print("Recuerdos recuperados:", len(result["recollections"]))
for r in result["recollections"]:
    print(f"  [{r.memory.memory_type}] {r.memory.content}")

print(f"\nMemorias activas: {result['total_memories']}")
print(f"Turno: {result['turn_count']}")

agent.end_session()
agent.close()
```

---

## 5. Providers soportados

| Provider | Env var | Modelo default |
|----------|---------|----------------|
| Qwen Cloud | `DASHSCOPE_API_KEY` | qwen-plus |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| OpenRouter | `OPENROUTER_API_KEY` | openai/gpt-4o |
| OpenAI | `OPENAI_API_KEY` | gpt-4o |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4 |

---

## 6. Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    Cliente                           │
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

Para LLM Connector:

```
┌─────────────────────────────────────────────────────┐
│              LLMConnector                            │
│                                                      │
│  User Input ──► MemoryAgent.perceive() ──► Memories  │
│       │                                              │
│       ▼                                              │
│  Build Context (system prompt + memories)            │
│       │                                              │
│       ▼                                              │
│  LLM API (DeepSeek / OpenAI / etc.)                  │
│       │                                              │
│       ▼                                              │
│  Response ──► MemoryAgent stores interaction         │
└─────────────────────────────────────────────────────┘
```
