"""Tests for algorithm performance baseline governance script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "governance"
    / "algorithm_performance_baseline.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "algorithm_performance_baseline_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load algorithm_performance_baseline module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_report(all_passed: bool) -> dict:
    return {
        "generated_at_utc": "2026-03-03T00:00:00+00:00",
        "summary": {
            "all_passed": all_passed,
            "checks_passed": 2 if all_passed else 1,
            "checks_total": 2,
        },
        "thresholds_ms": {
            "var_monte_carlo_p95": 250.0,
            "backtest_engine_p95": 1200.0,
        },
        "checks": {
            "var_monte_carlo_p95_ok": True,
            "backtest_engine_p95_ok": all_passed,
        },
        "metrics": {
            "var_monte_carlo": {
                "n": 8,
                "mean_ms": 100.0,
                "median_ms": 98.0,
                "p95_ms": 140.0,
                "p99_ms": 160.0,
                "min_ms": 90.0,
                "max_ms": 170.0,
            },
            "backtest_engine": {
                "n": 6,
                "mean_ms": 600.0,
                "median_ms": 580.0,
                "p95_ms": 1000.0 if all_passed else 1400.0,
                "p99_ms": 1100.0 if all_passed else 1500.0,
                "min_ms": 500.0,
                "max_ms": 1200.0 if all_passed else 1600.0,
            },
        },
    }


def test_main_generates_markdown_and_json_reports(tmp_path, monkeypatch):
    module = _load_module()
    output_md = tmp_path / "artifacts" / "algorithm-performance-baseline.md"
    output_json = tmp_path / "artifacts" / "algorithm-performance-baseline.json"
    monkeypatch.setattr(module, "_run_suite", lambda _args: _sample_report(all_passed=True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "algorithm_performance_baseline.py",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert output_md.exists()
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["summary"]["all_passed"] is True
    assert "Algorithm Performance Baseline" in output_md.read_text(encoding="utf-8")


def test_main_strict_returns_nonzero_on_failed_checks(tmp_path, monkeypatch):
    module = _load_module()
    output_md = tmp_path / "artifacts" / "algorithm-performance-baseline.md"
    output_json = tmp_path / "artifacts" / "algorithm-performance-baseline.json"
    monkeypatch.setattr(module, "_run_suite", lambda _args: _sample_report(all_passed=False))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "algorithm_performance_baseline.py",
            "--strict",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()
    assert exit_code == 2


def test_main_returns_nonzero_on_invalid_positive_numeric_args(tmp_path, monkeypatch):
    module = _load_module()
    output_md = tmp_path / "artifacts" / "algorithm-performance-baseline.md"
    output_json = tmp_path / "artifacts" / "algorithm-performance-baseline.json"
    run_called = {"value": False}

    def _should_not_run(_args):
        run_called["value"] = True
        return _sample_report(all_passed=True)

    monkeypatch.setattr(module, "_run_suite", _should_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "algorithm_performance_baseline.py",
            "--var-iterations",
            "0",
            "--backtest-p95-threshold-ms",
            "-1",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    assert run_called["value"] is False
    assert output_md.exists() is False
    assert output_json.exists() is False
