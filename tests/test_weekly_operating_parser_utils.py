"""Tests for weekly operating audit parser helper definitions."""

from __future__ import annotations

from scripts.governance.weekly_operating_parser_utils import (
    build_weekly_operating_argument_specs,
)


def test_build_weekly_operating_argument_specs_exposes_expected_defaults_and_flags():
    specs = build_weekly_operating_argument_specs()
    by_dest = {spec["dest"]: spec for spec in specs}

    assert by_dest["results_dir"]["default"] == "results"
    assert by_dest["pattern"]["default"] == "backtest*.json"
    assert by_dest["strict"]["action"] == "store_true"
    assert by_dest["strict_close"]["action"] == "store_true"
    assert by_dest["close_gate_only"]["action"] == "store_true"
    assert by_dest["performance_json"]["default"] == "artifacts/algorithm-performance-baseline.json"
    assert by_dest["latency_json"]["default"] == "artifacts/performance/latency_benchmark_report.json"
    assert by_dest["change_log_days"]["type"] is int
    assert "--inputs" in by_dest["inputs"]["flags"]
