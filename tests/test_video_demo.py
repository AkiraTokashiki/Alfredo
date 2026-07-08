"""Tests for the no-voice Devpost video demo script."""

from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path


def test_video_demo_script_runs_fast_with_large_caption_scenes():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "demo_video.py"

    result = subprocess.run(
        [sys.executable, str(script), "--fast"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    output = result.stdout
    assert "ALFREDO MEMORYAGENT" in output
    assert "NO VOICEOVER DEMO" in output
    assert "SESSION 1" in output
    assert "SESSION 2" in output
    assert "SESSION 3" in output
    assert "LARGE VAULT, SMALL PROMPT" in output
    assert "VAULT BENCHMARK" in output
    assert "BENCHMARK RESULTS" in output
    assert "DEVPOST TAKEAWAY" in output


def test_video_demo_planned_pauses_fit_two_minute_recording():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "demo_video.py"
    spec = importlib.util.spec_from_file_location("demo_video", script)
    assert spec is not None
    assert spec.loader is not None
    demo_video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo_video)

    assert 75 <= demo_video.planned_pause_seconds() <= 120
