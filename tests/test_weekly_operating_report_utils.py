"""Tests for weekly operating report assembly helpers."""

from __future__ import annotations

import pytest

from scripts.governance.weekly_operating_report_utils import (
    build_risk_exceptions,
    build_consistency_checks,
    build_operating_checklist,
    normalize_change_log,
    build_report_summary,
    normalize_regression_report,
    normalize_rollback_marker,
    normalize_optional_baseline_report,
    resolve_optional_report_check,
)


def test_build_consistency_checks_flags_threshold_breaches():
    checks = build_consistency_checks(
        latest_by_strategy={
            "A": {
                "source_file": "latest.json",
                "pnl": 220.0,
                "sharpe": 2.2,
                "max_drawdown_abs": 0.35,
            }
        },
        previous_by_strategy={
            "A": {
                "source_file": "previous.json",
                "pnl": 100.0,
                "sharpe": 1.0,
                "max_drawdown_abs": 0.10,
            }
        },
        thresholds={
            "max_abs_pnl_diff": 50.0,
            "max_abs_sharpe_diff": 0.5,
            "max_abs_max_drawdown_diff": 0.2,
        },
    )

    assert len(checks) == 1
    assert checks[0]["status"] == "FAIL"
    assert checks[0]["abs_pnl_diff"] == 120.0
    assert checks[0]["abs_sharpe_diff"] == pytest.approx(1.2)
    assert checks[0]["abs_max_drawdown_diff"] == pytest.approx(0.25)
    assert "abs_pnl_diff>50.0" in checks[0]["breached_rules"]
    assert "abs_sharpe_diff>0.5" in checks[0]["breached_rules"]
    assert "abs_max_drawdown_diff>0.2" in checks[0]["breached_rules"]


def test_optional_baseline_report_helpers_handle_missing_and_required_states():
    missing = normalize_optional_baseline_report(None)
    passing = normalize_optional_baseline_report({"executed": True, "summary": {"all_passed": True}})

    assert missing == {
        "executed": False,
        "summary": {"all_passed": None},
        "error": "",
        "path": "",
    }
    assert resolve_optional_report_check(missing, required=False) is None
    assert resolve_optional_report_check(missing, required=True) is False
    assert resolve_optional_report_check(passing, required=True) is True


def test_build_operating_checklist_and_summary_use_shared_inputs():
    snapshot_rows = [
        {"strategy": "A", "experiment_id": "EXP-1"},
        {"strategy": "B", "experiment_id": ""},
    ]
    risk_exceptions = [{"strategy": "B"}]
    consistency_checks = [{"strategy": "A", "status": "PASS"}]
    consistency_exceptions: list[dict[str, object]] = []
    checklist = build_operating_checklist(
        snapshot_rows=snapshot_rows,
        parse_errors=[],
        consistency_checks=consistency_checks,
        consistency_exceptions=consistency_exceptions,
        risk_exceptions=risk_exceptions,
        change_log={
            "executed": True,
            "shallow": False,
            "count": 2,
        },
        rollback_marker={"source": "tag", "tag": "backup-release-demo"},
        regression_result={"executed": True, "passed": True},
        performance_check=True,
        latency_check=None,
    )
    summary = build_report_summary(
        snapshot_rows=snapshot_rows,
        risk_exceptions=risk_exceptions,
        consistency_checks=consistency_checks,
        consistency_exceptions=consistency_exceptions,
        parse_errors=[],
    )

    assert checklist["kpi_snapshot_updated"] is True
    assert checklist["experiment_ids_assigned"] is False
    assert checklist["change_log_complete"] is True
    assert checklist["rollback_version_marked"] is True
    assert checklist["minimum_regression_passed"] is True
    assert checklist["performance_baseline_passed"] is True
    assert checklist["latency_baseline_passed"] is None
    assert checklist["anomalies_attributed"] is False
    assert summary == {
        "strategies": 2,
        "exceptions": 1,
        "consistency_pairs": 1,
        "consistency_exceptions": 0,
        "parse_errors": 0,
    }


def test_risk_exceptions_and_default_section_helpers_return_stable_shapes():
    snapshot_rows = [
        {
            "strategy": "A",
            "source_file": "a.json",
            "status": "PASS",
            "breached_rules": "",
        },
        {
            "strategy": "B",
            "source_file": "b.json",
            "status": "FAIL",
            "breached_rules": "sharpe<0.5",
        },
    ]

    assert build_risk_exceptions(snapshot_rows) == [
        {
            "strategy": "B",
            "source_file": "b.json",
            "breached_rules": "sharpe<0.5",
        }
    ]
    assert normalize_regression_report(None) == {
        "executed": False,
        "command": "",
        "passed": None,
        "return_code": None,
        "output_tail": "",
    }
    assert normalize_change_log(None) == {
        "executed": False,
        "since_days": 0,
        "entries": [],
        "count": 0,
        "shallow": False,
        "error": "",
    }
    assert normalize_rollback_marker(None) == {
        "executed": False,
        "tag": "",
        "error": "",
        "source": "",
    }
