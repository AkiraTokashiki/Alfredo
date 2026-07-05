# Hackathon Submission

## Track

**Track 1: MemoryAgent**

MemoryAgent matches Track 1 because it gives an AI agent persistent memory across sessions, remembers user preferences, retrieves critical memories under context limits, reinforces useful memories, and archives stale memories through forgetting curves.

## Project description

MemoryAgent is a persistent memory layer for AI agents running on Qwen Cloud. It stores user preferences, facts, and interaction history in SQLite, searches them with semantic embeddings, ranks recall candidates by semantic relevance, recency, importance, and recall strength, then injects the most relevant memories into the model context.

The agent improves across turns and sessions because useful memories are reinforced when retrieved, while low-strength or stale memories decay and are archived. The project includes a CLI, MCP integration, an LLM connector for Qwen Cloud's OpenAI-compatible API, demos, and tests for storage, retrieval, forgetting, and agent behavior.

## Required submission URLs

Fill these before final Devpost submission:

| Requirement | URL |
| --- | --- |
| Public code repository | `https://github.com/AkiraTokashiki/Alfredo` |
| Public demo video, about 3 minutes | `REPLACE_WITH_YOUTUBE_VIMEO_OR_FACEBOOK_VIDEO_URL` |
| Alibaba Cloud deployment proof recording | `REPLACE_WITH_PUBLIC_RECORDING_URL` |
| Alibaba Cloud proof code file | `https://github.com/AkiraTokashiki/Alfredo/blob/main/deploy/alibaba_cloud_proof.py` |
| Optional blog/social post | `REPLACE_WITH_BLOG_OR_SOCIAL_POST_URL` |

## Required repository files

- `LICENSE` — MIT License for open-source visibility at the repository root.
- `README.md` — setup, usage, tests, integrations, and hackathon demo instructions.
- `docs/ARCHITECTURE.md` — architecture diagram and system flow.
- `docs/architecture.svg` — rendered architecture diagram asset for judges.
- `deploy/alibaba_cloud_proof.py` — code that demonstrates Alibaba Cloud Function Compute and Qwen Cloud API usage.
- `examples/demo_basic.py` and `examples/demo_multi_session.py` — local functionality demos.

## Alibaba Cloud deployment proof instructions

The short deployment proof recording should show:

1. Alibaba Cloud console with the running backend resource, such as Function Compute or ECS.
2. Environment variables configured for the backend, including `DASHSCOPE_API_KEY`.
3. A terminal or console invocation of the deployed MemoryAgent backend.
4. A call to the proof script:

```bash
python deploy/alibaba_cloud_proof.py --region <region-id> --function-name <function-name>
```

5. Output confirming Alibaba Cloud Function Compute metadata and Qwen Cloud chat endpoint connectivity.

## Architecture diagram

See `docs/ARCHITECTURE.md` and the rendered diagram asset `docs/architecture.svg`.

## Demo video outline

Target length: about 3 minutes.

1. **0:00-0:20** — Introduce Track 1: MemoryAgent and the problem: agents forget user preferences across sessions.
2. **0:20-0:55** — Show MemoryAgent storing a preference in one session.
3. **0:55-1:25** — Start a new session and show cross-session recall.
4. **1:25-1:55** — Show retrieval scores, importance, strength, and forgetting behavior.
5. **1:55-2:25** — Show Qwen Cloud response using memory context.
6. **2:25-2:45** — Show Alibaba Cloud backend proof separately or point to the separate proof recording.
7. **2:45-3:00** — Summarize impact: persistent memory, timely forgetting, context-window-efficient recall.

## Suggested Devpost text description

MemoryAgent is a persistent memory system for Qwen Cloud agents. It lets an AI assistant accumulate experience across sessions, remember user preferences, retrieve critical memories within a limited context window, and forget stale or low-value information over time.

The system uses SQLite for durable storage, sentence-transformer embeddings for semantic search, multi-factor scoring for recall quality, Maximum Marginal Relevance for diversity, and an Ebbinghaus-inspired forgetting curve for decay and archival. The agent can run locally through the CLI, integrate with MCP clients, or connect to Qwen Cloud through an OpenAI-compatible LLM connector. The Alibaba Cloud deployment proof code demonstrates how the backend is verified on Alibaba Cloud and how Qwen Cloud APIs are used.

## Judging criteria mapping

| Criterion | How MemoryAgent addresses it |
| --- | --- |
| Technical Depth & Engineering (30%) | Uses Qwen Cloud's OpenAI-compatible API, MCP integration, SQLite WAL persistence, semantic embeddings, multi-factor retrieval scoring, Maximum Marginal Relevance, Ebbinghaus-inspired forgetting, reinforcement, and Alibaba Cloud deployment proof code. |
| Innovation & AI Creativity (30%) | Implements a modular long-term memory layer for agents with autonomous memory extraction, context-window-efficient recall, stale-memory archival, and cross-session preference recall. |
| Problem Value & Impact (25%) | Solves the real agent problem of forgetting user preferences and wasting prompt context on irrelevant history; can be reused by CLI agents, MCP clients, and Qwen Cloud applications. |
| Presentation & Documentation (15%) | Includes `README.md`, `docs/ARCHITECTURE.md`, this `SUBMISSION.md`, demo video outline, Alibaba Cloud proof instructions, and a root `LICENSE` file. |

### Technical Depth & Engineering details

- **Qwen Cloud usage**: `src/memory_agent/integrations/llm_connector.py` includes the `qwencloud` provider and sends memory-augmented chat requests to DashScope/Qwen's OpenAI-compatible endpoint.
- **Alibaba Cloud proof**: `deploy/alibaba_cloud_proof.py` uses Alibaba Cloud Function Compute SDK calls and Qwen Cloud API calls for the required backend deployment proof recording.
- **Custom agent memory stack**: `src/memory_agent/core/memory_store.py`, `src/memory_agent/core/retrieval.py`, and `src/memory_agent/core/forgetting.py` implement durable memory, semantic recall, reinforcement, and decay.
- **MCP integration**: `src/memory_agent/integrations/mcp_server.py` exposes memory operations to MCP-compatible clients.
- **Engineering verification**: tests cover storage, forgetting, retrieval, orchestrator behavior, QwenCloud provider wiring, and required submission artifacts.

#### Technical Depth answer for judges

MemoryAgent makes sophisticated use of QwenCloud by wrapping Qwen's OpenAI-compatible chat API with a persistent memory layer instead of sending stateless prompts. Before each QwenCloud call, the agent retrieves relevant long-term memories, formats them into bounded context, and sends the augmented conversation through the `qwencloud` provider. The project also exposes the same memory operations through MCP, so QwenCloud-backed agents and MCP-compatible clients can call memory tools such as perceive, search, store, stats, forget, and reinforce.

The engineering innovation is the custom memory pipeline: SQLite persistence with WAL mode, sentence-transformer embeddings, weighted semantic/recency/importance/strength scoring, Maximum Marginal Relevance for diverse recall, Ebbinghaus-inspired decay, and reinforcement on retrieval. This lets the agent keep useful preferences alive, reduce stale information, and avoid wasting limited context windows on duplicate or irrelevant memories.

### Innovation & AI Creativity details

- **Memory as an agent capability**: the project turns QwenCloud from a stateless chat model into a cross-session agent that accumulates experience.
- **Autonomous experience accumulation**: MemoryAgent extracts preferences, facts, and interaction summaries from normal conversation without requiring the user to manually save notes.
- **Creative forgetting model**: low-value memories decay while high-importance memories survive longer, so the agent can forget without deleting critical context.
- **Context-window awareness**: retrieval prioritizes memories that are semantically relevant, recent, important, strong, and diverse, instead of dumping the whole history into the prompt.
- **Reusable agent substrate**: the same memory engine works through CLI, MCP, custom Python, and QwenCloud-backed conversations.

#### Innovation answer for judges

The creative idea is not just storing chat logs. MemoryAgent models memory as a living system: experiences are extracted, scored, reinforced, decayed, retrieved, and archived. This gives QwenCloud-backed agents a practical long-term memory that behaves closer to human recall: important preferences persist, irrelevant details fade, and the model receives only the memories most likely to improve the current decision.

#### Architecture quality answer for judges

The architecture is modular by responsibility. Storage, embeddings, retrieval, forgetting, extraction, orchestration, MCP serving, and LLM provider calls are separate modules with narrow interfaces. This makes the system testable and lets the memory layer run through multiple front doors: CLI, MCP, custom Python, and QwenCloud-backed chat.

The scalability path is incremental: SQLite WAL is simple and reliable for local or single-backend deployments; embeddings are stored separately from memory records; retrieval scoring is configurable; MCP and HTTP-style integration make the backend deployable behind Alibaba Cloud Function Compute or ECS. The design can later swap SQLite for a managed database or vector backend without replacing the orchestrator or QwenCloud connector.

Error handling is explicit in the critical paths: embedding failures do not prevent memory storage, archived memories are soft-deleted instead of hard-deleted, API keys are read from environment variables, and `deploy/alibaba_cloud_proof.py` reports deployment-proof failures with structured JSON plus stderr diagnostics.

The non-trivial logic is in the memory scoring and lifecycle: semantic similarity, recency, importance, recall strength, MMR diversity, reinforcement, exponential decay, archival thresholds, session linking, and provider abstraction. The tech stack is intentionally boring where reliability matters and sophisticated where agent behavior benefits from it.

### Problem Value & Impact details

- **Real pain point**: AI agents forget user preferences between sessions, forcing users to repeat context and reducing trust.
- **Concrete value**: MemoryAgent lets QwenCloud agents remember durable preferences, facts, and useful prior interactions while forgetting stale details.
- **Productization path**: the memory layer can be embedded into assistants, coding agents, support bots, personal productivity agents, and MCP-enabled workflows.
- **Open-source reuse**: developers can use MemoryAgent as a standalone Python package, MCP server, or QwenCloud connector.
- **Scalability potential**: the current local SQLite backend is easy to deploy for demos and can evolve toward managed storage or vector infrastructure for larger teams.

#### Impact answer for judges

The impact is practical: long-running agents need durable, selective memory to feel useful after the first conversation. MemoryAgent solves that by making user preferences and important facts portable across sessions, while preventing old or low-value memories from polluting the prompt. This improves continuity, personalization, and decision quality for real QwenCloud applications.

#### Real-world relevance and scalability answer for judges

The authentic technical pain point is context loss. Stateless LLM apps force users to repeat preferences, constraints, identity facts, and prior decisions. Teams building agents often patch this with raw chat history, which grows quickly, wastes tokens, retrieves irrelevant details, and keeps outdated information alive. MemoryAgent provides a reusable solution: structured memory extraction, semantic retrieval, importance-aware scoring, reinforcement, and forgetting.

The business value applies to customer support assistants, developer agents, executive copilots, education tutors, healthcare intake assistants, and internal workflow agents where continuity matters. The open-source path is clear because the project exposes multiple integration surfaces: Python API for developers, CLI for demos, MCP server for agent clients, and QwenCloud provider support for production AI workflows. The backend can start with SQLite for simple deployment and later evolve to managed Alibaba Cloud databases or vector services without changing the agent-facing API.

### Presentation & Documentation details

- **README**: explains setup, demos, architecture overview, integrations, commands, QwenCloud usage, and hackathon submission links.
- **Architecture diagram**: `docs/ARCHITECTURE.md` contains a Mermaid diagram showing Qwen Cloud, Alibaba Cloud backend, MemoryAgent core, SQLite, embeddings, retrieval, and forgetting.
- **Submission guide**: `SUBMISSION.md` contains track, description, judging-criteria mapping, required URLs, proof-recording instructions, and demo-video outline.
- **Deployment proof code**: `deploy/alibaba_cloud_proof.py` gives judges a concrete code file showing Alibaba Cloud and QwenCloud API usage.
- **License**: root `LICENSE` file makes the repository visibly open source.
- **Demo path**: `examples/demo_basic.py` and `examples/demo_multi_session.py` demonstrate memory creation, recall, and persistence.

#### Presentation answer for judges

The submission is documented for both judges and developers. Judges can start at `SUBMISSION.md` for the track, description, video outline, architecture link, and deployment proof link. Developers can start at `README.md` for install and run commands. The architecture is visualized in `docs/ARCHITECTURE.md`, and the Alibaba Cloud proof requirement is backed by an actual repository code file rather than prose only.

## Public repository requirement

This local directory is not currently a Git repository. Before submission:

```bash
cd E:/CODE/MemoryAgent
git init
git add .
git commit -m "feat: prepare MemoryAgent hackathon submission"
git branch -M main
git remote add origin https://github.com/AkiraTokashiki/Alfredo.git
git push -u origin main
```

Then make the repository public and confirm the repository About panel detects the MIT license.
