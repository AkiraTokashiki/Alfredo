"""Alibaba Cloud deployment proof for MemoryAgent.

This script is intended for the hackathon deployment-proof recording. It checks
that the deployed backend can be inspected through Alibaba Cloud Function Compute
APIs and that the same environment can reach Qwen Cloud's OpenAI-compatible chat
API.

Required environment variables:
    ALIBABA_CLOUD_ACCESS_KEY_ID
    ALIBABA_CLOUD_ACCESS_KEY_SECRET
    DASHSCOPE_API_KEY

Example:
    python deploy/alibaba_cloud_proof.py \
        --region us-east-1 \
        --function-name memory-agent-backend \
        --qwen-message "Say MemoryAgent is deployed"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib import request

QWEN_CHAT_COMPLETIONS_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
DEFAULT_QWEN_MODEL = "qwen-plus"


@dataclass
class ProofResult:
    """Serializable deployment proof result."""

    ok: bool
    alibaba_cloud: dict[str, Any]
    qwen_cloud: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "alibaba_cloud": self.alibaba_cloud,
                "qwen_cloud": self.qwen_cloud,
            },
            indent=2,
            ensure_ascii=False,
        )


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def check_function_compute(region: str, function_name: str | None) -> dict[str, Any]:
    """Read Alibaba Cloud Function Compute metadata through the official SDK.

    The SDK dependency is optional for local development. Install it when
    recording deployment proof:

        pip install alibabacloud-fc20230330 alibabacloud-tea-openapi
    """
    access_key_id = _require_env("ALIBABA_CLOUD_ACCESS_KEY_ID")
    access_key_secret = _require_env("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    try:
        from alibabacloud_fc20230330.client import Client as FcClient
        from alibabacloud_fc20230330 import models as fc_models
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError as exc:
        raise RuntimeError(
            "Alibaba Cloud Function Compute SDK is not installed. "
            "Run: pip install alibabacloud-fc20230330 alibabacloud-tea-openapi"
        ) from exc

    endpoint = f"{region}.fc.aliyuncs.com"
    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=endpoint,
    )
    client = FcClient(config)

    if function_name:
        response = client.get_function(function_name)
        body = response.body.to_map() if hasattr(response.body, "to_map") else response.body
        return {
            "service": "Alibaba Cloud Function Compute",
            "region": region,
            "endpoint": endpoint,
            "function_name": function_name,
            "function": body,
        }

    request_obj = fc_models.ListFunctionsRequest(limit=10)
    response = client.list_functions(request_obj)
    body = response.body.to_map() if hasattr(response.body, "to_map") else response.body
    return {
        "service": "Alibaba Cloud Function Compute",
        "region": region,
        "endpoint": endpoint,
        "functions": body,
    }


def check_qwen_cloud(message: str, model: str) -> dict[str, Any]:
    """Call Qwen Cloud's OpenAI-compatible chat completions endpoint."""
    api_key = _require_env("DASHSCOPE_API_KEY")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are verifying a MemoryAgent backend deployment.",
            },
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        QWEN_CHAT_COMPLETIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8")
        data = json.loads(raw)

    return {
        "service": "Qwen Cloud DashScope OpenAI-compatible API",
        "endpoint": QWEN_CHAT_COMPLETIONS_URL,
        "model": model,
        "response_status": "ok",
        "response_preview": data["choices"][0]["message"]["content"][:240],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MemoryAgent Alibaba Cloud deployment.")
    parser.add_argument("--region", required=True, help="Alibaba Cloud region id, for example us-east-1")
    parser.add_argument("--function-name", default=None, help="Function Compute function name to inspect")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL, help="Qwen Cloud model name")
    parser.add_argument(
        "--qwen-message",
        default="Confirm MemoryAgent backend proof is running on Alibaba Cloud.",
        help="Message sent to Qwen Cloud during proof recording",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = ProofResult(
        ok=False,
        alibaba_cloud={},
        qwen_cloud={},
    )

    try:
        result.alibaba_cloud = check_function_compute(args.region, args.function_name)
        result.qwen_cloud = check_qwen_cloud(args.qwen_message, args.qwen_model)
        result.ok = True
        print(result.to_json())
        return 0
    except Exception as exc:
        result.ok = False
        print(result.to_json())
        print(f"deployment proof failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
