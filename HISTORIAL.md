# Historial del Proyecto — MemoryAgent

## 03/07/2026

### ✅ Feature completada
- (todo lo anterior)
- **MCP Server**: FastMCP server que expone 6 tools (memory__perceive, memory__search, memory__store, memory__stats, memory__forget, memory__reinforce) + 2 resources + 1 prompt. Soporta stdio y HTTP. Integrable con Hermes via `hermes mcp add`, con Claude Desktop, Cursor, etc.
- **LLM Connector**: `python -m memory_agent.integrations.llm_connector` — sesion interactiva con DeepSeek/OpenRouter/OpenAI/Anthropic usando MemoryAgent como memoria persistente. 4 providers soportados.
- **Guia de integracion**: `INTEGRACION.md` con ejemplos de configuracion para Hermes (MCP + skill), comandos de terminal, y uso programatico.

### ✅ Feature completada
- **Memory Store**: SQLite con WAL, 5 tablas (memories, embeddings, memory_tags, sessions, session_memories). CRUD completo, soft/hard delete, batch operations, keyword search fallback.
- **Embedding Engine**: sentence-transformers (all-MiniLM-L6-v2, 384d) con cache LRU, encode_multiple batch, cosine similarity.
- **Forgetting Curve**: Ebbinghaus exponencial con importancia modulada. Alta: ~90d, media: ~21d, baja: ~3d. Refuerzo en cada retrieval (+0.15 strength). Archival automatico bajo threshold.
- **Retrieval Engine**: Scoring combinado (semantico 40% + recencia 20% + importancia 20% + fuerza 20%). MMR diversity penalty para evitar duplicados semanticos.
- **Decision Engine**: Extraccion de patrones de preferencias y hechos del lenguaje natural (espanol + ingles).
- **Agent Orchestrator**: Ciclo completo perceive → extract → retrieve → decay. Sesiones multi-turn, consolidacion cada N turnos, vinculacion sesion-memoria.
- **CLI**: Click-based interactive chat con comandos /stats, /memories, /search, /forget.
- **Tests**: 43 tests, todos pasando en ~24s.
- **Demos**: demo_basic.py (sesion unica con 8 interacciones) y demo_multi_session.py (3 sesiones demostrando persistencia entre sesiones y olvido).

### 🔧 Fixes aplicados
- **Problema**: `sqlite3.Row` no tiene metodo `.get()` en Python 3.14. Se cambio a acceso por corchetes `row["column"]`.
- **Problema**: `get_sentence_embedding_dimension()` renombrado a `get_embedding_dimension()` en sentence-transformers 3.x. Warning corregido.
- **Problema**: test_decay_is_exponential usaba formula incorrecta mid_log. Se simplifico a `s1 * s1 ≈ s2`.
- **Problema**: test_top_k_limit y test_mmr_diversity timeout por crear EmbeddingEngine 20 veces. Se optimizo a encode_multiple batch.
- **Problema**: patron de extraccion "me gusta X" no matcheaba sin articulo. Se hizo el grupo opcional con `?`.

### ⚠️ Issues conocidos
- Los patrones de extraccion de preferencias pueden generar overlap en frases complejas. No critico porque el retrieval prioriza por score.
- La generacion de respuestas es template-based (sin LLM). Para uso real, conectar a un modelo.
- La demo basica muestra el modelo cargandose en la segunda iteracion (lazy load de sentence-transformers).

### ⚙️ Stack tecnologico
- Python 3.14.3
- SQLite (stdlib, WAL mode)
- sentence-transformers 5.6.0 (all-MiniLM-L6-v2)
- numpy, click, pytest
- Editable install via setuptools
