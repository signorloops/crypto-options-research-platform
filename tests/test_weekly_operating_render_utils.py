"""Tests for weekly operating audit markdown render helpers."""

from __future__ import annotations

from scripts.governance.weekly_operating_render_utils import (
    build_close_gate_markdown,
    build_weekly_operating_markdown,
)


def test_build_close_gate_markdown_renders_all_sections():
    report = {
        "generated_at_utc": "2026-03-06T00:00:00+00:00",
        "status": "FAIL",
        "gate": "READY_FOR_CLOSE",
        "reason": "status=PENDING_MANUAL_SIGNOFF",
        "signoff_status": "PENDING_MANUAL_SIGNOFF",
        "signoff_json": "artifacts/weekly-signoff-pack.json",
        "auto_blockers": ["performance baseline passed"],
        "pending_items": ["灰度发布完成"],
        "manual_missing": ["灰度发布完成"],
        "role_signoffs_missing": ["Risk 签字"],
        "pr_brief": "### Weekly Close Gate\n- Status: FAIL",
    }

    markdown = build_close_gate_markdown(report)

    assert "# Weekly Close Gate" in markdown
    assert "- [ ] performance baseline passed" in markdown
    assert "- [ ] 灰度发布完成" in markdown
    assert "- [ ] Risk 签字" in markdown
    assert "```markdown" in markdown
    assert "### Weekly Close Gate" in markdown


def test_build_weekly_operating_markdown_renders_benchmarks_and_parse_errors():
    report = {
        "generated_at_utc": "2026-03-06T00:00:00+00:00",
        "inputs": ["a.json"],
        "summary": {
            "strategies": 1,
            "exceptions": 0,
            "consistency_pairs": 1,
            "consistency_exceptions": 0,
        },
        "kpi_snapshot": [
            {
                "strategy": "Stable",
                "source_file": "a.json",
                "experiment_id": "EXP-1",
                "pnl": 100.0,
                "sharpe": 1.2,
                "max_drawdown_abs": 0.05,
                "var_breach_rate": 0.0,
                "fill_calibration_error": 0.0,
                "status": "PASS",
            }
        ],
        "risk_exceptions": [],
        "consistency_checks": [
            {
                "strategy": "Stable",
                "latest_source_file": "a.json",
                "previous_source_file": "b.json",
                "abs_pnl_diff": 12.0,
                "abs_sharpe_diff": 0.1,
                "abs_max_drawdown_diff": 0.01,
                "status": "PASS",
                "breached_rules": "",
            }
        ],
        "checklist": {
            "kpi_snapshot_updated": True,
            "experiment_ids_assigned": True,
            "risk_thresholds_confirmed": True,
            "change_log_complete": True,
            "rollback_version_marked": True,
            "minimum_regression_passed": True,
            "performance_baseline_passed": True,
            "latency_baseline_passed": False,
            "consistency_check_completed": True,
            "risk_exception_report_output": True,
            "anomalies_attributed": True,
        },
        "regression": {"executed": False},
        "performance_baseline": {
            "executed": True,
            "summary": {"all_passed": True, "checks_passed": 2, "checks_total": 2},
            "metrics": {
                "var_monte_carlo": {"p95_ms": 1.2},
                "backtest_engine": {"p95_ms": 4.5},
            },
        },
        "latency_baseline": {
            "executed": True,
            "summary": {"all_passed": False, "checks_passed": 5, "checks_total": 6},
            "benchmarks": [
                {"name": "Quote Generation", "p95_ms": 4.2, "target_ms": 10.0, "meets_target": True}
            ],
        },
        "change_log": {
            "executed": True,
            "since_days": 7,
            "shallow": False,
            "entries": [{"date": "2026-03-06", "commit": "abc12345", "subject": "demo"}],
        },
        "rollback_marker": {"executed": True, "tag": "backup-release-demo", "source": "tag"},
        "manual_checklist_items": ["灰度发布完成", "ADR"],
        "incomplete_tasks": ["延迟基线达标", "灰度发布完成"],
        "parse_errors": [{"file": "bad.json", "error": "invalid"}],
    }

    markdown = build_weekly_operating_markdown(report)

    assert "# Weekly Operating Audit" in markdown
    assert "## KPI Snapshot" in markdown
    assert "Quote Generation" in markdown
    assert "backup-release-demo" in markdown
    assert "- [ ] 延迟基线达标" in markdown
    assert "## Parse Errors" in markdown
    assert "bad.json" in markdown
