"""Standalone LLM connector: MemoryAgent + OpenAI-compatible API.

Connects MemoryAgent's persistent memory to any LLM with an OpenAI-compatible
API (DeepSeek, OpenRouter, Anthropic, etc.).

Usage:
    # Set API key
    set DEEPSEEK_API_KEY=sk-...     # Windows CMD
    export DEEPSEEK_API_KEY=sk-...  # bash

    # Run with DeepSeek
    python -m memory_agent.llm --provider deepseek --model deepseek-chat

    # Run with OpenRouter
    python -m memory_agent.llm --provider openrouter --model openai/gpt-4o
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.core.config import MemoryAgentConfig

# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, str]] = {
    "qwencloud": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
    },
}

SYSTEM_PROMPT = """Eres MemoryAgent, un asistente con memoria persistente.

Tienes acceso a un sistema de memoria que:
1. RECUERDA preferencias, hechos, y experiencias de conversaciones pasadas
2. OLVIDA gradualmente información no relevante (curva de Ebbinghaus)
3. RECUPERA recuerdos relevantes usando búsqueda semántica

Cuando respondas:
- Usa la memoria contextual que se te provee en cada turno
- Si el usuario menciona una preferencia, confírmala antes de guardarla
- Si no estás seguro de algo, sé honesto
- Responde en el mismo idioma que el usuario

Formato de la memoria contextual:
[RECUERDOS]:
  - [tipo] contenido (score: X.XX)
[/RECUERDOS]"""


class LLMConnector:
    """Connects MemoryAgent to an OpenAI-compatible LLM API."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        db_path: str | Path | None = None,
        system_prompt: str | None = None,
    ):
        if provider not in PROVIDERS:
            available = ", ".join(PROVIDERS.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

        self.provider = provider
        pconf = PROVIDERS[provider]
        self.model = model or pconf["default_model"]
        self.base_url = pconf["base_url"]
        self.api_key = os.getenv(pconf["api_key_env"], "")

        if not self.api_key:
            print(
                f"  [!] No {pconf['api_key_env']} set. LLM calls will fail.",
                file=sys.stderr,
            )

        # Initialize MemoryAgent
        config = MemoryAgentConfig.default()
        self.agent = MemoryAgent(config=config, db_path=db_path)

        # Conversation history (for multi-turn context)
        self.messages: list[dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        self.http = httpx.Client(timeout=60.0)

    # ------------------------------------------------------------------
    # Memory context
    # ------------------------------------------------------------------

    def _build_memory_context(self, user_input: str) -> str:
        """Retrieve relevant memories and format them for the LLM."""
        results = self.agent.retrieval.retrieve(user_input, top_k=5, use_mmr=True)
        if not results:
            return ""

        lines = ["[RECUERDOS]:"]
        for r in results:
            mem = r.memory
            type_icon = {"episodic": "📝", "semantic": "💡", "preference": "❤️", "procedural": "🔧"}
            icon = type_icon.get(mem.memory_type, "📌")
            lines.append(
                f"  {icon} [{mem.memory_type}] {mem.content} (score: {r.score:.2f})"
            )
        lines.append("[/RECUERDOS]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM API and return the response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        resp = self.http.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Main turn
    # ------------------------------------------------------------------

    def turn(self, user_input: str) -> str:
        """Process one user turn: retrieve memories + call LLM + store memory.

        Returns the LLM's response.
        """
        # 1. Retrieve memories
        memory_context = self._build_memory_context(user_input)

        # 2. Build messages
        turn_messages = list(self.messages)

        if memory_context:
            # Inject memories as system context
            turn_messages.append({"role": "system", "content": memory_context})

        turn_messages.append({"role": "user", "content": user_input})

        # 3. Call LLM
        response = self._call_llm(turn_messages)

        # 4. Store memories
        self.agent.perceive(user_input, response)

        # 5. Update conversation history (keep last 20 for context)
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response})
        if len(self.messages) > 21:  # system + 10 exchanges
            # Keep system + last 20 messages (10 exchanges)
            self.messages = [self.messages[0]] + self.messages[-20:]

        return response

    def close(self) -> None:
        close = getattr(self.http, "close", None)
        if close is not None:
            close()
        self.agent.close()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------


def run_interactive(
    provider: str = "deepseek",
    model: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Run an interactive chat session with memory."""
    connector = LLMConnector(
        provider=provider,
        model=model,
        db_path=db_path,
        system_prompt=SYSTEM_PROMPT,
    )
    connector.agent.init_session(f"llm-{provider}")

    # Welcome
    pconf = PROVIDERS[provider]
    print(f"\n  MemoryAgent + {provider.upper()} ({connector.model})")
    print(f"  Base: {connector.base_url}")
    print(f"  DB:   {connector.agent.db_path}")
    print(f"  Modelo: {connector.model}")
    print(f"  Comandos: /stats, /search <q>, /memories, /quit")
    print("=" * 55)

    try:
        while True:
            try:
                user_input = input("\n  Tu > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Handle simple commands
            if user_input.startswith("/"):
                cmd = user_input.split(maxsplit=1)
                command = cmd[0].lower()
                arg = cmd[1] if len(cmd) > 1 else ""

                if command == "/quit":
                    break
                elif command == "/stats":
                    stats = connector.agent.get_stats()
                    print(f"\n  📊 Stats:")
                    for k, v in stats.items():
                        print(f"     {k}: {v}")
                    continue
                elif command == "/search" and arg:
                    results = connector.agent.retrieval.retrieve(arg, top_k=5)
                    for r in results:
                        print(f"  #{r.memory.id} [{r.memory.memory_type}] {r.memory.content[:70]}")
                    continue
                elif command == "/memories":
                    mems = connector.agent.store.get_all_active_memories()
                    for m in mems[:10]:
                        print(f"  #{m.id} [{m.memory_type}] {m.content[:70]}")
                    continue
                else:
                    print(f"  Comandos: /stats, /search <q>, /memories, /quit")
                    continue

            # Process turn
            print(f"  [pensando...]", end="", flush=True)
            try:
                response = connector.turn(user_input)
                print(f"\r", end="", flush=True)
                print(f"\n  Agent > {response}")
            except Exception as e:
                print(f"\r  [Error: {e}]")

            # Show memory stats line
            stats = connector.agent.get_stats()
            print(f"  [memorias: {stats['total_active']} activas | "
                  f"{stats['session_turns']} turnos]")

    finally:
        connector.close()
        print("\n  Sesion terminada.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MemoryAgent + LLM")
    parser.add_argument("--provider", "-p", default="deepseek", choices=list(PROVIDERS.keys()))
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--db", default=None, help="Path to memory DB")
    parser.add_argument("--query", "-q", default=None, help="Single query (non-interactive)")

    args = parser.parse_args()

    if args.query:
        connector = LLMConnector(
            provider=args.provider,
            model=args.model,
            db_path=args.db,
            system_prompt=SYSTEM_PROMPT,
        )
        connector.agent.init_session("llm-single")
        response = connector.turn(args.query)
        print(response)
        connector.close()
    else:
        run_interactive(
            provider=args.provider,
            model=args.model,
            db_path=args.db,
        )
