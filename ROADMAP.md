# Roadmap

Alfredo is currently a local SDK and CLI for a namespace-aware memory layer. The shipped surface is the Python package, `alfredo` command, `python -m memory_agent` compatibility path, local SQLite vault, deterministic offline mode, optional semantic embeddings, and MCP stdio/HTTP adapters. See [`INTEGRATION.md`](INTEGRATION.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the supported behavior today.

## Current local SDK

Now:

- Keep the local-first SQLite lifecycle reliable and explainable: perceive, extract, validate/trust, store, retrieve, pack context, reinforce, supersede, decay/archive.
- Preserve namespace isolation, provider/dimension guards, deterministic offline embeddings, selected/dropped IDs, evidence, and explicit forget semantics.
- Keep the offline CLI, Python API, MCP recipes, synthetic benchmark, focused documentation contracts, and Windows-safe contributor workflow reproducible.

These are maintenance priorities for the current local SDK, not a promise of a hosted service or a release date.

## Future work (planned, not current commitments)

The following ideas are intentionally future work. They have no shipped API, schedule, availability guarantee, or hosting commitment:

- **Dashboard (future):** an optional local inspection UI for namespaces, evidence, selected/dropped IDs, archive state, and retention actions. It must not turn the core into a hosted SaaS requirement.
- **Hosting (future):** deployment guidance or a separately secured hosted service may be explored only with explicit authentication, tenant isolation, encryption, retention, backup, and operational review. Current users should run their own process and SQLite vault.
- **TypeScript (future):** a TypeScript client or protocol bindings may be explored after the Python contracts and MCP behavior stabilize. No TypeScript package is promised by this roadmap.

Future work remains subject to maintainer review, threat modeling, implementation capacity, and compatibility testing. Until those changes land in the repository and are documented as supported, treat them as proposals rather than features.

## Links

- [Integration guide](INTEGRATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
