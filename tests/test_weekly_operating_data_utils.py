"""Tests for weekly operating audit data extraction helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.governance.weekly_operating_data_utils import (
    collect_strategy_snapshots,
    extract_strategy_rows,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_extract_strategy_rows_uses_multiple_payload_maps_and_infers_experiment_id(tmp_path):
    source = tmp_path / "backtest_results_demo.json"
    rows = extract_strategy_rows(
        {
            "A": {
                "summary": {"total_pnl": 100.0, "experiment_id": "EXP-1"},
                "metrics": {"sharpe": 1.2, "max_drawdown": -0.1},
                "risk": {"var_breach_rate": 0.02},
            },
            "B": {
                "metrics": {"final_pnl": 20.0, "fill_error": 0.03},
                "risk": {"max_dd": 0.05},
            },
            "ignored": "not-a-mapping",
        },
        source,
    )

    assert rows == [
        {
            "strategy": "A",
            "source_file": "backtest_results_demo.json",
            "pnl": 100.0,
            "sharpe": 1.2,
            "max_drawdown_abs": 0.1,
            "var_breach_rate": 0.02,
            "fill_calibration_error": None,
            "experiment_id": "EXP-1",
        },
        {
            "strategy": "B",
            "source_file": "backtest_results_demo.json",
            "pnl": 20.0,
            "sharpe": None,
            "max_drawdown_abs": 0.05,
            "var_breach_rate": None,
            "fill_calibration_error": 0.03,
            "experiment_id": "AUTO-backtest_results_demo",
        },
    ]


def test_collect_strategy_snapshots_tracks_latest_previous_and_parse_errors(tmp_path):
    older = tmp_path / "results" / "older.json"
    newer = tmp_path / "results" / "newer.json"
    invalid = tmp_path / "results" / "invalid.json"
    _write(older, json.dumps({"Stable": {"summary": {"total_pnl": 50.0, "sharpe_ratio": 1.0}}}))
    _write(newer, json.dumps({"Stable": {"summary": {"total_pnl": 80.0, "sharpe_ratio": 1.2}}}))
    _write(invalid, "{not-json")
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))
    os.utime(newer, (newer.stat().st_atime, newer.stat().st_mtime + 10))

    latest, previous, parse_errors = collect_strategy_snapshots([older, newer, invalid])

    assert latest["Stable"]["source_file"] == "newer.json"
    assert latest["Stable"]["pnl"] == 80.0
    assert previous["Stable"]["source_file"] == "older.json"
    assert previous["Stable"]["pnl"] == 50.0
    assert parse_errors[0]["file"].endswith("invalid.json")
