"""Tests for weekly operating KPI/risk audit automation."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
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
    assert report["checklist"]["rollback_marker_from_tag"] is True
    assert "异常项已归因" in report["incomplete_tasks"]


def test_build_report_treats_commit_fallback_and_shallow_log_as_incomplete(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_001b.json"
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
    assert report["checklist"]["rollback_version_marked"] is False
    assert report["checklist"]["rollback_marker_from_tag"] is False


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


def test_main_executes_regression_cmd_without_shell(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_003.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 100.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.1,
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
        if isinstance(cmd, list) and cmd[:3] == ["git", "rev-parse", "--is-shallow-repository"]:
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


def test_run_regression_command_returns_report_and_ignores_blank(tmp_path, monkeypatch):
    module = _load_module()
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="line-1\nline-2\n",
            stderr="line-3\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._run_regression_command("", tmp_path) is None

    report = module._run_regression_command(
        "python -m pytest -q tests/test_pricing_inverse.py",
        tmp_path,
    )

    assert report == {
        "executed": True,
        "command": "python -m pytest -q tests/test_pricing_inverse.py",
        "passed": False,
        "return_code": 1,
        "output_tail": "line-1\nline-2\n\nline-3",
    }
    regression_cmd, regression_kwargs = calls[0]
    assert regression_cmd == ["python", "-m", "pytest", "-q", "tests/test_pricing_inverse.py"]
    assert regression_kwargs["cwd"] == tmp_path
    assert regression_kwargs.get("shell") in (None, False)


def test_resolve_input_files_prefers_explicit_inputs_before_discovery(tmp_path, monkeypatch):
    module = _load_module()
    explicit = tmp_path / "explicit.json"
    discovered = tmp_path / "results" / "backtest_results.json"
    _write(explicit, "{}")
    _write(discovered, "{}")
    calls: list[tuple[Path, str]] = []

    def fake_discover(results_dir: Path, pattern: str) -> list[Path]:
        calls.append((results_dir, pattern))
        return [discovered]

    monkeypatch.setattr(module, "_discover_input_files", fake_discover)

    resolved_explicit = module._resolve_input_files(
        repo_root=tmp_path,
        explicit_inputs=[str(explicit)],
        results_dir="results",
        pattern="backtest*.json",
    )
    resolved_discovered = module._resolve_input_files(
        repo_root=tmp_path,
        explicit_inputs=[],
        results_dir="results",
        pattern="backtest*.json",
    )

    assert resolved_explicit == [explicit.resolve()]
    assert resolved_discovered == [discovered]
    assert calls == [((tmp_path / "results").resolve(), "backtest*.json")]


def test_resolve_audit_paths_builds_resolved_output_locations(tmp_path):
    module = _load_module()

    paths = module._resolve_audit_paths(
        repo_root=tmp_path,
        output_md="artifacts/weekly-operating-audit.md",
        output_json="artifacts/weekly-operating-audit.json",
        signoff_json="artifacts/weekly-signoff-pack.json",
        close_gate_output_md="artifacts/weekly-close-gate.md",
        close_gate_output_json="artifacts/weekly-close-gate.json",
    )

    assert paths["output_md"] == (tmp_path / "artifacts/weekly-operating-audit.md").resolve()
    assert paths["output_json"] == (tmp_path / "artifacts/weekly-operating-audit.json").resolve()
    assert paths["signoff_json"] == (tmp_path / "artifacts/weekly-signoff-pack.json").resolve()
    assert paths["close_gate_output_md"] == (tmp_path / "artifacts/weekly-close-gate.md").resolve()
    assert paths["close_gate_output_json"] == (
        tmp_path / "artifacts/weekly-close-gate.json"
    ).resolve()


def test_load_baseline_reports_reads_performance_and_latency_json(tmp_path, monkeypatch):
    module = _load_module()
    calls: list[tuple[Path, str]] = []

    def fake_load(path: Path, missing_error: str):
        calls.append((path, missing_error))
        return {"path": str(path), "error": missing_error, "executed": True}

    monkeypatch.setattr(module, "load_optional_report", fake_load)

    reports = module._load_baseline_reports(
        repo_root=tmp_path,
        performance_json="artifacts/perf.json",
        latency_json="artifacts/latency.json",
    )

    assert reports["performance"]["path"].endswith("artifacts/perf.json")
    assert reports["latency"]["path"].endswith("artifacts/latency.json")
    assert calls == [
        ((tmp_path / "artifacts/perf.json").resolve(), "missing_performance_json"),
        ((tmp_path / "artifacts/latency.json").resolve(), "missing_latency_json"),
    ]


def test_report_issue_messages_returns_exit_only_when_strict(capsys):
    module = _load_module()

    strict_exit = module._report_issue_messages(["problem-1", "problem-2"], strict=True)
    relaxed_exit = module._report_issue_messages(["problem-1"], strict=False)
    empty_exit = module._report_issue_messages([], strict=True)

    captured = capsys.readouterr()
    assert "problem-1" in captured.out
    assert captured.out.count("problem-1") == 2
    assert "problem-2" not in captured.out
    assert strict_exit == 2
    assert relaxed_exit is None
    assert empty_exit is None


def test_handle_strict_close_returns_exit_when_gate_not_ready(tmp_path, monkeypatch):
    module = _load_module()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "_evaluate_close_gate",
        lambda path: (False, "status=PENDING_MANUAL_SIGNOFF", {"status": "PENDING_MANUAL_SIGNOFF"}),
    )
    monkeypatch.setattr(
        module,
        "_write_close_gate_report",
        lambda **kwargs: calls.append(kwargs),
    )

    exit_code = module._handle_strict_close(
        strict_close=True,
        signoff_json_path=tmp_path / "signoff.json",
        close_gate_md=tmp_path / "close.md",
        close_gate_json=tmp_path / "close.json",
    )

    assert exit_code == 2
    assert calls and calls[0]["close_detail"] == "status=PENDING_MANUAL_SIGNOFF"


def test_run_close_gate_only_returns_status_and_writes_report(tmp_path, monkeypatch):
    module = _load_module()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_evaluate_close_gate",
        lambda path: (True, "READY_FOR_CLOSE", {"status": "READY_FOR_CLOSE"}),
    )
    monkeypatch.setattr(module, "_write_close_gate_report", lambda **kwargs: calls.append(kwargs))

    exit_code = module._run_close_gate_only(
        strict_close=True,
        signoff_json_path=tmp_path / "signoff.json",
        close_gate_md=tmp_path / "close.md",
        close_gate_json=tmp_path / "close.json",
    )

    assert exit_code == 0
    assert calls and calls[0]["close_ready"] is True


def test_prepare_audit_run_loads_thresholds_inputs_and_supporting_reports(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "demo.json"
    _write(result_path, "{}")

    monkeypatch.setattr(module, "_load_thresholds", lambda path: {"min_sharpe": 0.5})
    monkeypatch.setattr(
        module,
        "_load_consistency_thresholds",
        lambda path: {"max_abs_pnl_diff": 10.0},
    )
    monkeypatch.setattr(module, "_resolve_input_files", lambda **kwargs: [result_path])
    monkeypatch.setattr(
        module,
        "_run_regression_command",
        lambda command, repo_root: {"executed": True, "passed": True, "command": command},
    )
    monkeypatch.setattr(module, "_collect_recent_changes", lambda repo_root, days: {"count": days})
    monkeypatch.setattr(module, "_detect_latest_tag", lambda repo_root: {"tag": "backup-release-demo"})
    monkeypatch.setattr(
        module,
        "_load_baseline_reports",
        lambda **kwargs: {"performance": {"executed": True}, "latency": {"executed": True}},
    )

    payload = module._prepare_audit_run(
        repo_root=tmp_path,
        args=Namespace(
            thresholds="config/t.json",
            consistency_thresholds="config/c.json",
            inputs=[str(result_path)],
            results_dir="results",
            pattern="backtest*.json",
            regression_cmd="pytest -q",
            change_log_days=7,
            performance_json="artifacts/perf.json",
            latency_json="artifacts/latency.json",
        ),
    )

    assert payload["thresholds"] == {"min_sharpe": 0.5}
    assert payload["consistency_thresholds"] == {"max_abs_pnl_diff": 10.0}
    assert payload["input_files"] == [result_path]
    assert payload["regression_result"]["command"] == "pytest -q"
    assert payload["change_log"] == {"count": 7}
    assert payload["rollback_marker"] == {"tag": "backup-release-demo"}
    assert payload["baselines"]["performance"]["executed"] is True
    assert payload["baselines"]["latency"]["executed"] is True


def test_detect_latest_tag_falls_back_to_commit_when_head_is_not_tagged(tmp_path):
    module = _load_module()

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Test User")
    _git("config", "user.email", "test@example.com")
    _write(tmp_path / "README.md", "demo\n")
    _git("add", "README.md")
    _git("commit", "-m", "init")
    _git("tag", "-a", "backup-release-20260306-demo", "-m", "rollback baseline")
    _write(tmp_path / "next.txt", "next\n")
    _git("add", "next.txt")
    _git("commit", "-m", "next")

    marker = module._detect_latest_tag(tmp_path)

    assert marker["executed"] is True
    assert marker["source"] == "commit"
    assert marker["tag"].startswith("HEAD-")


def test_discover_input_files_searches_nested_result_directories(tmp_path):
    module = _load_module()
    nested = tmp_path / "results" / "backtest_full_history" / "backtest_full_nested.json"
    _write(nested, json.dumps({"S": {"summary": {"total_pnl": 1.0, "sharpe_ratio": 1.0}}}))

    found = module._discover_input_files(tmp_path / "results", "backtest*.json")

    assert nested in found


def test_build_report_flags_consistency_exceptions(tmp_path):
    module = _load_module()
    older = tmp_path / "results" / "backtest_results_old.json"
    newer = tmp_path / "results" / "backtest_results_new.json"
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


def test_build_report_requires_performance_baseline_when_requested(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_perf_required.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 100.0,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": -0.1,
                    }
                }
            }
        ),
    )

    report = module._build_report(
        [result_path],
        dict(module.DEFAULT_THRESHOLDS),
        performance_result={
            "executed": False,
            "summary": {"all_passed": None},
            "error": "missing_performance_json",
        },
        performance_required=True,
    )

    assert report["checklist"]["performance_baseline_passed"] is False
    assert "性能基线达标" in report["incomplete_tasks"]


def test_build_report_does_not_require_performance_by_default(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_perf_optional.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 110.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.09,
                    }
                }
            }
        ),
    )

    report = module._build_report(
        [result_path],
        dict(module.DEFAULT_THRESHOLDS),
        performance_result={
            "executed": False,
            "summary": {"all_passed": None},
            "error": "missing_performance_json",
        },
        performance_required=False,
    )

    assert report["checklist"]["performance_baseline_passed"] is None
    assert "性能基线达标" not in report["incomplete_tasks"]


def test_build_report_requires_latency_baseline_when_requested(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_latency_required.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 101.0,
                        "sharpe_ratio": 1.1,
                        "max_drawdown": -0.08,
                    }
                }
            }
        ),
    )

    report = module._build_report(
        [result_path],
        dict(module.DEFAULT_THRESHOLDS),
        latency_result={
            "executed": False,
            "summary": {"all_passed": None},
            "error": "missing_latency_json",
        },
        latency_required=True,
    )

    assert report["checklist"]["latency_baseline_passed"] is False
    assert "延迟基线达标" in report["incomplete_tasks"]


def test_build_report_does_not_require_latency_by_default(tmp_path):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_latency_optional.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 111.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.07,
                    }
                }
            }
        ),
    )

    report = module._build_report(
        [result_path],
        dict(module.DEFAULT_THRESHOLDS),
        latency_result={
            "executed": False,
            "summary": {"all_passed": None},
            "error": "missing_latency_json",
        },
        latency_required=False,
    )

    assert report["checklist"]["latency_baseline_passed"] is None
    assert "延迟基线达标" not in report["incomplete_tasks"]


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
                        "total_pnl": 3500.0,
                        "sharpe_ratio": 1.8,
                        "max_drawdown": -0.35,
                    }
                }
            }
        ),
    )
    os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))
    os.utime(newer, (newer.stat().st_atime, newer.stat().st_mtime + 10))

    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
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
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["summary"]["consistency_exceptions"] == 1


def test_main_require_performance_strict_fails_when_json_missing(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_perf_missing.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 120.0,
                        "sharpe_ratio": 1.1,
                        "max_drawdown": -0.08,
                    }
                }
            }
        ),
    )
    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"
    missing_performance = tmp_path / "artifacts" / "missing-performance.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    _write(consistency_thresholds_path, json.dumps(module.DEFAULT_CONSISTENCY_THRESHOLDS))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(result_path),
            "--thresholds",
            str(thresholds_path),
            "--consistency-thresholds",
            str(consistency_thresholds_path),
            "--performance-json",
            str(missing_performance),
            "--require-performance",
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["checklist"]["performance_baseline_passed"] is False
    assert report["performance_baseline"]["error"] == "missing_performance_json"


def test_main_require_latency_strict_fails_when_json_missing(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_latency_missing.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 120.0,
                        "sharpe_ratio": 1.1,
                        "max_drawdown": -0.08,
                    }
                }
            }
        ),
    )
    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"
    missing_latency = tmp_path / "artifacts" / "missing-latency.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    _write(consistency_thresholds_path, json.dumps(module.DEFAULT_CONSISTENCY_THRESHOLDS))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(result_path),
            "--thresholds",
            str(thresholds_path),
            "--consistency-thresholds",
            str(consistency_thresholds_path),
            "--latency-json",
            str(missing_latency),
            "--require-latency",
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["checklist"]["latency_baseline_passed"] is False
    assert report["latency_baseline"]["error"] == "missing_latency_json"


def test_main_require_latency_passes_when_baseline_passed(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_latency_ok.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 90.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.06,
                    }
                }
            }
        ),
    )
    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    latency_path = tmp_path / "artifacts" / "latency-benchmark.json"
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    _write(consistency_thresholds_path, json.dumps(module.DEFAULT_CONSISTENCY_THRESHOLDS))
    _write(
        latency_path,
        json.dumps(
            {
                "summary": {"all_passed": True, "checks_passed": 2, "checks_total": 2},
                "benchmarks": [
                    {"name": "Quote Generation", "p95_ms": 4.2, "target_ms": 35.0},
                    {"name": "End-to-End", "p95_ms": 65.0, "target_ms": 100.0},
                ],
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(result_path),
            "--thresholds",
            str(thresholds_path),
            "--consistency-thresholds",
            str(consistency_thresholds_path),
            "--latency-json",
            str(latency_path),
            "--require-latency",
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["checklist"]["latency_baseline_passed"] is True
    assert report["latency_baseline"]["executed"] is True
    assert report["latency_baseline"]["summary"]["all_passed"] is True


def test_main_require_performance_passes_when_baseline_passed(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_perf_ok.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 90.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.06,
                    }
                }
            }
        ),
    )
    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    performance_path = tmp_path / "artifacts" / "algorithm-performance-baseline.json"
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    _write(consistency_thresholds_path, json.dumps(module.DEFAULT_CONSISTENCY_THRESHOLDS))
    _write(
        performance_path,
        json.dumps(
            {
                "summary": {"all_passed": True, "checks_passed": 2, "checks_total": 2},
                "metrics": {
                    "var_monte_carlo": {"p95_ms": 1.2},
                    "backtest_engine": {"p95_ms": 5.6},
                },
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(result_path),
            "--thresholds",
            str(thresholds_path),
            "--consistency-thresholds",
            str(consistency_thresholds_path),
            "--performance-json",
            str(performance_path),
            "--require-performance",
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["checklist"]["performance_baseline_passed"] is True
    assert report["performance_baseline"]["executed"] is True
    assert report["performance_baseline"]["summary"]["all_passed"] is True


def test_main_require_performance_strict_fails_when_json_invalid(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_perf_invalid.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 95.0,
                        "sharpe_ratio": 1.05,
                        "max_drawdown": -0.07,
                    }
                }
            }
        ),
    )
    thresholds_path = tmp_path / "thresholds.json"
    consistency_thresholds_path = tmp_path / "consistency_thresholds.json"
    performance_path = tmp_path / "artifacts" / "algorithm-performance-baseline.json"
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    _write(consistency_thresholds_path, json.dumps(module.DEFAULT_CONSISTENCY_THRESHOLDS))
    _write(performance_path, "{invalid-json")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--inputs",
            str(result_path),
            "--thresholds",
            str(thresholds_path),
            "--consistency-thresholds",
            str(consistency_thresholds_path),
            "--performance-json",
            str(performance_path),
            "--require-performance",
            "--output-md",
            str(report_md),
            "--output-json",
            str(report_json),
            "--strict",
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["checklist"]["performance_baseline_passed"] is False
    assert report["performance_baseline"]["executed"] is False
    assert report["performance_baseline"]["error"]
    assert report["performance_baseline"]["error"] != "missing_performance_json"


def test_main_close_gate_only_strict_fails_when_signoff_missing(tmp_path, monkeypatch):
    module = _load_module()
    missing_signoff = tmp_path / "missing-signoff.json"
    close_md = tmp_path / "artifacts" / "weekly-close-gate.md"
    close_json = tmp_path / "artifacts" / "weekly-close-gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--close-gate-only",
            "--strict-close",
            "--signoff-json",
            str(missing_signoff),
            "--close-gate-output-md",
            str(close_md),
            "--close-gate-output-json",
            str(close_json),
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    assert close_json.exists()
    report = json.loads(close_json.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["reason"] == "missing_signoff_json"
    assert "### Weekly Close Gate" in report["pr_brief"]
    assert "Status: FAIL" in report["pr_brief"]


def test_main_close_gate_only_strict_passes_when_ready(tmp_path, monkeypatch):
    module = _load_module()
    signoff_json = tmp_path / "weekly-signoff-pack.json"
    close_md = tmp_path / "artifacts" / "weekly-close-gate.md"
    close_json = tmp_path / "artifacts" / "weekly-close-gate.json"
    _write(signoff_json, json.dumps({"status": "READY_FOR_CLOSE"}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_operating_audit.py",
            "--close-gate-only",
            "--strict-close",
            "--signoff-json",
            str(signoff_json),
            "--close-gate-output-md",
            str(close_md),
            "--close-gate-output-json",
            str(close_json),
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    report = json.loads(close_json.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["signoff_status"] == "READY_FOR_CLOSE"
    assert "Status: PASS" in report["pr_brief"]


def test_close_gate_report_lists_actionable_auto_blocker_remediations():
    module = _load_module()
    signoff_json = Path("/tmp/demo-signoff.json")
    report = module._build_close_gate_report(
        signoff_json_path=signoff_json,
        close_ready=False,
        close_detail="status=AUTO_BLOCKED",
        signoff_payload={
            "status": "AUTO_BLOCKED",
            "auto_blockers": [
                "performance baseline passed",
                "latency baseline passed",
                "rollback_baseline_not_tagged",
            ],
            "pending_items": [],
            "manual_items": [],
            "role_signoffs": [],
        },
    )

    assert "Rerun algorithm performance baseline and fix regressions." in report["action_items"]
    assert "Rerun latency benchmark and reduce latency regressions." in report["action_items"]
    assert (
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate."
        in report["action_items"]
    )
    assert "Next actions:" in report["pr_brief"]


def test_close_gate_report_maps_current_signoff_auto_check_labels_to_actions():
    module = _load_module()
    report = module._build_close_gate_report(
        signoff_json_path=Path("/tmp/demo-signoff-current.json"),
        close_ready=False,
        close_detail="status=AUTO_BLOCKED",
        signoff_payload={
            "status": "AUTO_BLOCKED",
            "auto_blockers": [
                "rollback version marked",
                "canary recommendation is PROCEED_CANARY",
                "decision is APPROVE_CANARY",
            ],
            "pending_items": [],
            "manual_items": [],
            "role_signoffs": [],
        },
    )

    assert (
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate."
        in report["action_items"]
    )
    assert (
        "Resolve canary blockers until recommendation becomes PROCEED_CANARY."
        in report["action_items"]
    )
    assert (
        "Resolve decision blockers until decision becomes APPROVE_CANARY." in report["action_items"]
    )


def test_main_strict_close_fails_when_signoff_not_ready(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "results" / "backtest_results_ready.json"
    _write(
        result_path,
        json.dumps(
            {
                "Stable": {
                    "summary": {
                        "total_pnl": 100.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.1,
                    }
                }
            }
        ),
    )
    thresholds_path = tmp_path / "thresholds.json"
    signoff_json = tmp_path / "weekly-signoff-pack.json"
    close_md = tmp_path / "artifacts" / "weekly-close-gate.md"
    close_json = tmp_path / "artifacts" / "weekly-close-gate.json"
    report_md = tmp_path / "artifacts" / "weekly-operating-audit.md"
    report_json = tmp_path / "artifacts" / "weekly-operating-audit.json"
    _write(thresholds_path, json.dumps(module.DEFAULT_THRESHOLDS))
    _write(signoff_json, json.dumps({"status": "PENDING_MANUAL_SIGNOFF"}))
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
            "--strict-close",
            "--signoff-json",
            str(signoff_json),
            "--close-gate-output-md",
            str(close_md),
            "--close-gate-output-json",
            str(close_json),
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    assert report_json.exists()
    close_report = json.loads(close_json.read_text(encoding="utf-8"))
    assert close_report["status"] == "FAIL"
    assert close_report["reason"] == "status=PENDING_MANUAL_SIGNOFF"
    assert close_report["action_items"]
    assert "Next actions:" in close_report["pr_brief"]
