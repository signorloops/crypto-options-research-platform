"""Tests for weekly PnL attribution report generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "weekly_pnl_attribution.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("weekly_pnl_attribution_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_pnl_attribution module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_extracts_attribution_fields(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_001.json"
    _write(
        result_path,
        json.dumps(
            {
                "Strategy-A": {
                    "summary": {
                        "experiment_id": "EXP-001",
                        "spread_capture": 100.0,
                        "adverse_selection_cost": 20.0,
                        "inventory_cost": 5.0,
                        "hedging_cost": 7.0,
                    }
                },
                "Strategy-B": {
                    "metrics": {
                        "spread_capture": 88.0,
                    }
                },
            }
        ),
    )

    report = module._build_report([result_path])

    assert report["summary"]["strategies"] == 2
    assert report["summary"]["missing_entries"] == 1
    rows = {row["strategy"]: row for row in report["attribution_snapshot"]}
    assert rows["Strategy-A"]["experiment_id"] == "EXP-001"
    assert rows["Strategy-A"]["missing_fields"] == []
    assert rows["Strategy-B"]["spread_capture"] == 88.0
    assert "hedging_cost" in rows["Strategy-B"]["missing_fields"]


def test_main_strict_exits_nonzero_when_no_input_files(tmp_path, monkeypatch):
    module = _load_module()
    report_md = tmp_path / "artifacts" / "weekly-pnl-attribution.md"
    report_json = tmp_path / "artifacts" / "weekly-pnl-attribution.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_pnl_attribution.py",
            "--results-dir",
            str(tmp_path / "missing-results"),
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    assert not report_md.exists()
    assert not report_json.exists()
