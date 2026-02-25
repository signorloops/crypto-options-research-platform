"""Tests for weekly operating KPI/risk audit automation."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
            "shallow": False,
            "error": "",
        },
        rollback_marker={"executed": True, "tag": "v0.1.0", "error": "", "source": "tag"},
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


def test_build_report_treats_commit_fallback_and_shallow_log_as_incomplete(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_003.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 88.0,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": -0.05,
                        "experiment_id": "EXP-003",
                    }
                }
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
            "shallow": True,
            "error": "",
        },
        rollback_marker={"executed": True, "tag": "HEAD-abc12345", "error": "", "source": "commit"},
    )

    assert report["checklist"]["change_log_complete"] is False
    assert report["checklist"]["rollback_version_marked"] is True
    assert report["checklist"]["rollback_marker_from_tag"] is False


def test_main_regression_command_runs_without_shell(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_004.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 100.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.10,
                    }
                }
            }
        ),
    )

    thresholds_path = tmp_path / "thresholds.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"

    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        if (
            isinstance(cmd, list)
            and cmd[:2] == ["git", "rev-parse"]
            and "--is-shallow-repository" in cmd
        ):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="false\n", stderr="")
        if isinstance(cmd, list) and cmd[:2] == ["git", "log"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if isinstance(cmd, list) and cmd[:3] == ["git", "describe", "--tags"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="v0.1.0\n", stderr="")
        if isinstance(cmd, list) and cmd[:3] == ["git", "rev-parse", "--short"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="abc12345\n", stderr=""
            )

        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
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
            "--regression-cmd",
            "python -m pytest -q tests/test_pricing_inverse.py",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    regression_cmd, regression_kwargs = calls[0]
    assert regression_cmd == ["python", "-m", "pytest", "-q", "tests/test_pricing_inverse.py"]
    assert regression_kwargs.get("shell") in (None, False)


def test_discover_input_files_searches_nested_result_directories(tmp_path):
    module = _load_module()
    nested = tmp_path / "results" / "backtest_full_history" / "backtest_full_nested.json"
    _write(nested, json.dumps({"S": {"summary": {"total_pnl": 1.0, "sharpe_ratio": 1.0}}}))

    found = module._discover_input_files(tmp_path / "results", "backtest*.json")

    assert nested in found


def test_build_report_flags_consistency_exceptions(tmp_path):
    module = _load_module()
    newer = tmp_path / "results" / "backtest_results_new.json"
    older = tmp_path / "results" / "backtest_results_old.json"
    _write(
        older,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 100.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.10,
                    }
                }
            }
        ),
    )
    _write(
        newer,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 4000.0,
                        "sharpe_ratio": 1.9,
                        "max_drawdown": -0.30,
                    }
                }
            }
        ),
    )
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))
    os.utime(newer, (newer.stat().st_atime, newer.stat().st_mtime + 10))

    report = module._build_report(
        [newer, older],
        dict(module.DEFAULT_THRESHOLDS),
        consistency_thresholds={
            "max_abs_pnl_diff": 100.0,
            "max_abs_sharpe_diff": 0.2,
            "max_abs_max_drawdown_diff": 0.05,
        },
    )

    assert report["summary"]["consistency_exceptions"] == 1
    assert report["checklist"]["consistency_check_completed"] is False
    assert report["consistency_checks"][0]["status"] == "FAIL"


def test_main_strict_exits_nonzero_when_consistency_exceptions_exist(tmp_path, monkeypatch):
    module = _load_module()
    older = tmp_path / "results" / "backtest_results_old2.json"
    newer = tmp_path / "results" / "backtest_results_new2.json"
    _write(
        older,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 100.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.10,
                    }
                }
            }
        ),
    )
    _write(
        newer,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 5000.0,
                        "sharpe_ratio": 2.0,
                        "max_drawdown": -0.25,
                    }
                }
            }
        ),
    )
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))
    os.utime(newer, (newer.stat().st_atime, newer.stat().st_mtime + 10))

    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    _write(
        thresholds_path,
        json.dumps(
            {
                "min_sharpe": 0.1,
                "max_abs_drawdown": 1.0,
                "max_var_breach_rate": 1.0,
                "max_fill_calibration_error": 1.0,
            }
        ),
    )
    _write(
        consistency_thresholds_path,
        json.dumps(
            {
                "max_abs_pnl_diff": 100.0,
                "max_abs_sharpe_diff": 0.2,
                "max_abs_max_drawdown_diff": 0.05,
            }
        ),
    )

    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(newer),
            str(older),
            "--thresholds",
            str(thresholds_path),
            "--consistency-thresholds",
            str(consistency_thresholds_path),
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    json_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert json_report["summary"]["exceptions"] == 0
    assert json_report["summary"]["consistency_exceptions"] == 1
