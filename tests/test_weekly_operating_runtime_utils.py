"""Tests for weekly operating audit runtime/load helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.governance.weekly_operating_runtime_utils import (
    load_optional_report,
    load_threshold_map,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_threshold_map_uses_defaults_and_validates_numbers(tmp_path):
    defaults = {"a": 1.0, "b": 2.0}
    path = tmp_path / "thresholds.json"
    _write(path, json.dumps({"a": "3.5"}))

    loaded = load_threshold_map(path, defaults, label="threshold")

    assert loaded == {"a": 3.5, "b": 2.0}


def test_load_threshold_map_raises_on_invalid_numeric_value(tmp_path):
    path = tmp_path / "thresholds.json"
    _write(path, json.dumps({"a": "bad"}))

    with pytest.raises(ValueError, match="Invalid threshold value for 'a'"):
        load_threshold_map(path, {"a": 1.0}, label="threshold")


def test_load_optional_report_handles_missing_invalid_and_valid_payloads(tmp_path):
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    valid = tmp_path / "valid.json"
    _write(invalid, "{bad-json")
    _write(valid, json.dumps({"summary": {"all_passed": True}}))

    missing_report = load_optional_report(missing, missing_error="missing_perf")
    invalid_report = load_optional_report(invalid, missing_error="missing_perf")
    valid_report = load_optional_report(valid, missing_error="missing_perf")

    assert missing_report == {
        "executed": False,
        "summary": {"all_passed": None},
        "error": "missing_perf",
        "path": str(missing.resolve()),
    }
    assert invalid_report["executed"] is False
    assert invalid_report["error"]
    assert invalid_report["path"] == str(invalid.resolve())
    assert valid_report == {
        "summary": {"all_passed": True},
        "executed": True,
        "path": str(valid.resolve()),
        "error": "",
    }
