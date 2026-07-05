"""Click-based CLI for MemoryAgent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig


@click.group()
@click.option("--db", default=None, help="Path to SQLite database file")
@click.option(
    "--model",
    default=None,
    help="Embedding model name (sentence-transformers)",
)
@click.pass_context
def cli(ctx: click.Context, db: str | None, model: str | None) -> None:
    """MemoryAgent — persistent memory for AI agents.

    An agent that accumulates experience autonomously, remembers
    user preferences, and retrieves critical memories within
    limited context windows.
    """
    ctx.ensure_object(dict)

    # Resolve db path
    if db:
        db_path = Path(db).resolve()
    else:
        db_path = Path.cwd() / "memory_agent.db"

    config = MemoryAgentConfig.default()
    if model:
        config.embedding.model_name = model

    agent = MemoryAgent(config=config, db_path=db_path)

    # Don't auto-init session; let subcommands handle it
    ctx.obj["agent"] = agent


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@cli.command()
@click.option("--label", "-l", default="", help="Session label")
@click.pass_context
def chat(ctx: click.Context, label: str) -> None:
    """Start an interactive chat session."""
    agent: MemoryAgent = ctx.obj["agent"]

    print("\n  MemoryAgent — Interactive Session")
    print("  Commands: /stats, /memories, /search <q>, /forget <id>, /help, /quit")
    print("=" * 50)

    agent.init_session(label=label or None)

    try:
        while True:
            try:
                user_input = input("\n  Tu > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                _handle_command(agent, user_input)
                continue

            # Process through memory cycle
            result = agent.perceive(user_input)

            # Print recollections
            if result["recollection_text"]:
                print(f"\n  {'─' * 40}")
                print(result["recollection_text"])
                print(f"  {'─' * 40}")

            # Print new memories
            if result["new_memories"]:
                for mem in result["new_memories"]:
                    print(f"  [+] {mem.memory_type}: {mem.content[:60]}...")

            # Print stats line
            if result["archived"] > 0:
                print(f"  [archivados: {result['archived']}]")

            # Generate a response
            response = _generate_response(agent, user_input, result)
            print(f"\n  Agent > {response}")

    finally:
        agent.end_session()
        print("\n  Session ended.")


def _handle_command(agent: MemoryAgent, cmd: str) -> None:
    """Handle a slash command."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/quit":
        raise KeyboardInterrupt()

    elif command == "/help":
        print("""
  Commands:
    /stats        — Show memory statistics
    /memories     — List all active memories
    /search <q>   — Semantic search memories
    /forget <id>  — Delete a memory by ID
    /help         — This help
    /quit         — Exit session
        """.strip())

    elif command == "/stats":
        stats = agent.get_stats()
        print(f"\n  Memory Statistics:")
        print(f"  {'─' * 40}")
        print(f"  Active memories:   {stats['total_active']}")
        print(f"  Archived:          {stats['archived']}")
        print(f"  Embeddings:        {stats['embedding_count']}")
        print(f"  Session turns:     {stats['session_turns']}")
        print(f"  Avg importance:    {stats['avg_importance']}")
        print(f"\n  By type:")
        for t, c in stats["type_distribution"].items():
            print(f"    {t}: {c}")
        print(f"\n  Decay lifespans:")
        for level, days in stats["decay_lifespans_days"].items():
            print(f"    {level}: ~{days} days")

    elif command == "/memories":
        memories = agent.store.get_all_active_memories()
        if not memories:
            print("  No memories.")
            return
        print(f"\n  Active memories ({len(memories)}):")
        print(f"  {'─' * 40}")
        for m in memories[:20]:
            tags = f" [{', '.join(m.tags[:2])}]" if m.tags else ""
            print(f"  #{m.id}: [{m.memory_type}] {m.content[:70]}{tags}")
            print(f"       importancia={m.importance:.1f} fuerza={m.strength:.2f} "
                  f"accesos={m.access_count}")

    elif command == "/search":
        if not arg:
            print("  Use: /search <query>")
            return
        results = agent.retrieval.retrieve(arg, top_k=5)
        if not results:
            print("  No results.")
            return
        print(f"\n  Search results for '{arg}':")
        print(f"  {'─' * 40}")
        for r in results:
            m = r.memory
            print(f"  #{m.id} [{m.memory_type}] (score={r.score:.3f})")
            print(f"       {m.content[:80]}")

    elif command == "/forget":
        if not arg or not arg.isdigit():
            print("  Use: /forget <memory_id>")
            return
        mid = int(arg)
        agent.store.delete_memory(mid)
        print(f"  Memory #{mid} deleted.")

    else:
        print(f"  Unknown command: {command}. Type /help")


def _generate_response(
    agent: MemoryAgent, user_input: str, result: dict
) -> str:
    """Generate a response based on the user input and recollections.

    In a full deployment, this would call an LLM. For this demo,
    we use a template-based response.
    """
    recollections = result["recollection_text"]
    new_count = len(result["new_memories"])
    total = result["total_memories"]

    # Simple response template
    response_parts = []

    # Acknowledge new memories
    if new_count > 0:
        response_parts.append(f"Entendido. He guardado {new_count} {'nuevo recuerdo' if new_count == 1 else 'nuevos recuerdos'}.")

    # Reference recollections
    if recollections:
        top = result["recollections"][0]
        response_parts.append(
            f"Recorde que {top.memory.content[:50].lower()}..."
        )

    # Final note
    response_parts.append(
        f"Ya tengo {total} {'recuerdo' if total == 1 else 'recuerdos'} en mi memoria persistente."
    )

    return " ".join(response_parts)


# ------------------------------------------------------------------
# Additional CLI commands
# ------------------------------------------------------------------


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show memory statistics without starting a session."""
    agent: MemoryAgent = ctx.obj["agent"]
    stats_data = agent.get_stats()

    click.echo("\n📊 MemoryAgent Statistics")
    click.echo("━" * 40)
    click.echo(f"  Active memories:   {stats_data['total_active']}")
    click.echo(f"  Archived:          {stats_data['archived']}")
    click.echo(f"  Embeddings:        {stats_data['embedding_count']}")
    click.echo(f"  Avg importance:    {stats_data['avg_importance']}")
    click.echo()
    click.echo("  By type:")
    for t, c in stats_data["type_distribution"].items():
        click.echo(f"    {t}: {c}")
    click.echo()
    click.echo("  Decay lifespans:")
    for level, days in stats_data["decay_lifespans_days"].items():
        click.echo(f"    {level}: ~{days} days")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int) -> None:
    """Semantic memory search."""
    agent: MemoryAgent = ctx.obj["agent"]
    results = agent.retrieval.retrieve(query, top_k=top_k)

    if not results:
        click.echo("  No matching memories found.")
        return

    click.echo(f"\n  Search results for: '{query}'")
    click.echo("━" * 50)
    for i, r in enumerate(results, 1):
        m = r.memory
        click.echo(
            f"  {i}. #{m.id} [{m.memory_type}] "
            f"(score={r.score:.3f}, imp={m.importance:.1f})"
        )
        click.echo(f"     {m.content[:80]}")
        click.echo()


@cli.command()
@click.pass_context
def memories(ctx: click.Context) -> None:
    """List all active memories."""
    agent: MemoryAgent = ctx.obj["agent"]
    all_memories = agent.store.get_all_active_memories()

    if not all_memories:
        click.echo("  No memories stored.")
        return

    click.echo(f"\n  Active Memories ({len(all_memories)})")
    click.echo("━" * 50)
    for m in all_memories:
        click.echo(
            f"  #{m.id:<4} [{m.memory_type:<10}] "
            f"imp={m.importance:.2f} str={m.strength:.2f} "
            f"acc={m.access_count}"
        )
        click.echo(f"      {m.content[:80]}")
        click.echo()


@cli.command()
@click.option("--http", is_flag=True, help="Run in HTTP mode instead of stdio")
@click.option("--host", default="localhost", help="HTTP host (default: localhost)")
@click.option("--port", default=8090, type=int, help="HTTP port (default: 8090)")
@click.pass_context
def mcp(ctx: click.Context, http: bool, host: str, port: int) -> None:
    """Run MemoryAgent as an MCP server.

    Exposes memory tools via Model Context Protocol. Use stdio mode (default)
    for Hermes or Claude Desktop integration. Use --http for remote clients.
    """
    from memory_agent.integrations.mcp_server import run_mcp_server
    run_mcp_server(host=host if http else None, port=port if http else None)


@cli.command()
@click.option("--provider", "-p", default="qwencloud",
              help="LLM provider: qwencloud, deepseek, openrouter, openai, anthropic")
@click.option("--model", "-m", default=None, help="Model name override")
@click.option("--query", "-q", default=None, help="Single query (non-interactive)")
@click.pass_context
def llm(ctx: click.Context, provider: str, model: str | None, query: str | None) -> None:
    """Chat with Qwen Cloud or another LLM using MemoryAgent memory.

    Requires the corresponding API key env var to be set.
    """
    from memory_agent.integrations.llm_connector import run_interactive, LLMConnector

    if query:
        connector = LLMConnector(
            provider=provider, model=model,
            db_path=ctx.obj.get("db_path"),
            system_prompt="You are a helpful assistant with persistent memory.",
        )
        connector.agent.init_session("llm-single")
        response = connector.turn(query)
        print(response)
        connector.close()
    else:
        run_interactive(provider=provider, model=model)
