"""Tests for release candidate readiness guard."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.governance.release_candidate_guard import evaluate_release_candidate


def _write_pyproject(path: Path, *, version: str, classifiers: list[str]) -> None:
    classifier_lines = "\n".join(f'    "{classifier}",' for classifier in classifiers)
    path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "corp"',
                f'version = "{version}"',
                "classifiers = [",
                classifier_lines,
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_signoff(
    path: Path,
    *,
    status: str,
    auto_blockers: list[str] | None = None,
    pending_items: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "auto_blockers": auto_blockers or [],
                "pending_items": pending_items or [],
            }
        ),
        encoding="utf-8",
    )


def test_evaluate_release_candidate_fails_on_alpha_metadata(tmp_path: Path):
    pyproject_path = tmp_path / "pyproject.toml"
    signoff_path = tmp_path / "weekly-signoff-pack.json"

    _write_pyproject(
        pyproject_path,
        version="0.1.0",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3.9",
        ],
    )
    _write_signoff(signoff_path, status="READY_FOR_CLOSE")

    report = evaluate_release_candidate(
        pyproject_path=pyproject_path,
        signoff_path=signoff_path,
    )

    assert report["passed"] is False
    assert "project metadata is still alpha" in report["failures"]


def test_evaluate_release_candidate_passes_when_metadata_and_signoff_are_ready(tmp_path: Path):
    pyproject_path = tmp_path / "pyproject.toml"
    signoff_path = tmp_path / "weekly-signoff-pack.json"

    _write_pyproject(
        pyproject_path,
        version="0.2.0rc1",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3.9",
        ],
    )
    _write_signoff(signoff_path, status="READY_FOR_CLOSE")

    report = evaluate_release_candidate(
        pyproject_path=pyproject_path,
        signoff_path=signoff_path,
    )

    assert report["passed"] is True
    assert report["failures"] == []
