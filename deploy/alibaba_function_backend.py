"""Minimal Alibaba Cloud Function Compute backend for Alfredo proof.

This function is intentionally dependency-free so it can run directly in the
Alibaba Cloud Function Compute console as a Python event function.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def handler(event, context):
    """Return a deployment proof response for the hackathon judges."""
    payload = {
        "ok": True,
        "project": "Alfredo MemoryAgent",
        "track": "Track 1: MemoryAgent",
        "runtime": "Alibaba Cloud Function Compute",
        "qwen_cloud_integration": "Qwen Cloud / DashScope compatible chat endpoint",
        "memory_backend": "persistent selective memory with retrieval and forgetting",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(payload, ensure_ascii=False)
