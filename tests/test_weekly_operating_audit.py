"""Tests for weekly operating KPI/risk audit automation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "governance" / "weekly_operating_audit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("weekly_operating_audit_test_module", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly_operating_audit module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_report_extracts_snapshot_and_flags_exceptions(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_001.json"
    _write(
        result_path,
        json.dumps(
            {
                "Strategy-A": {
                    "summary": {
                        "total_pnl": 1200.0,
                        "sharpe_ratio": 1.4,
                        "max_drawdown": -0.12,
                        "experiment_id": "EXP-001",
                    }
                },
                "Strategy-B": {
                    "metrics": {
                        "final_pnl": -50.0,
                        "sharpe": 0.1,
                        "max_drawdown": 0.31,
                    },
                    "summary": {"experiment_id": "EXP-002"},
                },
            }
        ),
    )

    report = module._build_report(
        [result_path],
        dict(module.DEFAULT_THRESHOLDS),
        change_log={
            "executed": True,
            "since_days": 7,
            "entries": [{"date": "2026-02-25", "commit": "abc12345", "subject": "demo"}],
            "count": 1,
            "error": "",
        },
        rollback_marker={"executed": True, "tag": "v0.1.0", "error": ""},
    )

    assert report["summary"]["strategies"] == 2
    assert report["summary"]["exceptions"] == 1
    assert report["risk_exceptions"][0]["strategy"] == "Strategy-B"
    assert "sharpe<" in report["risk_exceptions"][0]["breached_rules"]
    assert report["checklist"]["kpi_snapshot_updated"] is True
    assert report["checklist"]["experiment_ids_assigned"] is True
    assert report["checklist"]["change_log_complete"] is True
    assert report["checklist"]["rollback_version_marked"] is True
    assert "异常项已归因" in report["incomplete_tasks"]


def test_main_strict_exits_nonzero_when_exceptions_exist(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_002.json"
    _write(
        result_path,
        json.dumps(
            {
                "Risky": {
                    "summary": {
                        "total_pnl": 10.0,
                        "sharpe_ratio": 0.01,
                        "max_drawdown": -0.5,
                    }
                }
            }
        ),
    )

    thresholds_path = tmp_path / "thresholds.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(result_path),
            "--thresholds",
            str(thresholds_path),
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    assert report_md.exists()
    assert report_json.exists()
    md_text = report_md.read_text(encoding="utf-8")
    json_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert "Weekly Operating Audit" in md_text
    assert json_report["summary"]["exceptions"] == 1
    assert json_report["checklist"]["experiment_ids_assigned"] is True
    assert json_report["kpi_snapshot"][0]["experiment_id"].startswith("AUTO-")
