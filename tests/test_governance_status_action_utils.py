"""Tests for shared governance status/remediation helpers."""

from __future__ import annotations

from scripts.governance.status_action_utils import (
    build_close_gate_action_items,
    build_incomplete_tasks,
    remediation_for_canary_blockers,
)


def test_build_incomplete_tasks_applies_optional_baseline_requirements():
    checklist = {
        "kpi_snapshot_updated": False,
        "experiment_ids_assigned": True,
        "change_log_complete": False,
        "rollback_version_marked": True,
        "minimum_regression_passed": None,
        "performance_baseline_passed": False,
        "latency_baseline_passed": True,
        "consistency_check_completed": False,
        "anomalies_attributed": False,
    }

    tasks = build_incomplete_tasks(
        checklist,
        performance_required=True,
        latency_required=False,
        manual_items=["灰度发布完成", "ADR"],
    )

    assert tasks == [
        "KPI 快照更新",
        "变更记录完整",
        "最小回归通过",
        "性能基线达标",
        "一致性检查完成",
        "异常项已归因",
        "灰度发布完成",
        "ADR",
    ]


def test_remediation_for_canary_blockers_dedupes_and_preserves_order():
    actions = remediation_for_canary_blockers(
        [
            "rollback_baseline_not_tagged",
            "performance_baseline_failed",
            "rollback_baseline_not_tagged",
        ]
    )

    assert actions == [
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate",
        "Rerun algorithm performance baseline and fix regressions",
    ]


def test_build_close_gate_action_items_prioritizes_blockers_then_manual_work():
    actions = build_close_gate_action_items(
        signoff_status="AUTO_BLOCKED",
        auto_blockers=[
            "performance baseline passed",
            "rollback version marked",
        ],
        manual_missing=["灰度发布完成"],
        role_signoffs_missing=["Risk 签字"],
        close_ready=False,
    )

    assert actions == [
        "Rerun algorithm performance baseline and fix regressions.",
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate.",
        "Complete remaining manual checks.",
        "Collect all role sign-offs (Research/Engineering/Risk).",
    ]
