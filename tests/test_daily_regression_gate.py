"""Tests for daily regression gate automation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "daily_regression_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("daily_regression_gate_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load daily_regression_gate module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_returns_zero_when_all_commands_pass(tmp_path, monkeypatch):
    module = _load_module()
    output_md = tmp_path / "daily.md"
    output_json = tmp_path / "daily.json"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daily_regression_gate.py",
            "--cmd",
            "python -m pytest -q tests/test_volatility.py",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["summary"]["all_passed"] is True


def test_main_strict_returns_nonzero_on_failed_command(tmp_path, monkeypatch):
    module = _load_module()
    output_md = tmp_path / "daily.md"
    output_json = tmp_path / "daily.json"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=2, stdout="", stderr="failed\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daily_regression_gate.py",
            "--cmd",
            "python -m pytest -q tests/test_volatility.py",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["summary"]["all_passed"] is False
    assert report["summary"]["failed"] == 1
