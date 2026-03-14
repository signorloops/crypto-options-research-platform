"""Tests for project release metadata."""

from __future__ import annotations

from pathlib import Path

import tomli


def test_project_metadata_reflects_release_candidate_state():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        project = tomli.load(fh)["project"]

    assert project["version"] == "0.2.0rc1"
    classifiers = set(project["classifiers"])
    assert "Development Status :: 3 - Alpha" not in classifiers
    assert "Development Status :: 4 - Beta" in classifiers
