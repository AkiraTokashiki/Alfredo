"""Tests for hackathon submission artifacts."""

from __future__ import annotations

from pathlib import Path


REQUIRED_SUBMISSION_ARTIFACTS = [
    Path("LICENSE"),
    Path("README.md"),
    Path("docs/ARCHITECTURE.md"),
    Path("SUBMISSION.md"),
    Path("deploy/alibaba_cloud_proof.py"),
]


def test_hackathon_submission_artifacts_are_present():
    """Track 1 submission should include the required repository artifacts."""
    repo_root = Path(__file__).resolve().parents[1]

    missing = [
        artifact.as_posix()
        for artifact in REQUIRED_SUBMISSION_ARTIFACTS
        if not (repo_root / artifact).is_file()
    ]

    assert missing == []
