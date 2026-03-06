#!/usr/bin/env python3
"""Shared governance status, task, and remediation helpers."""

from __future__ import annotations

from typing import Any

MANUAL_CHECKLIST_ITEMS: list[str] = [
    "灰度发布完成",
    "24h 观察完成",
    "是否触发回滚已决策",
    "收益归因表",
    "变更与回滚记录",
    "ADR",
]

_CANARY_BLOCKER_REMEDIATIONS: dict[str, str] = {
    "performance_baseline_failed": "Rerun algorithm performance baseline and fix regressions",
    "latency_baseline_failed": "Rerun latency benchmark and reduce latency regressions",
    "rollback_baseline_not_tagged": (
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate"
    ),
    "minimum_regression_failed": "Fix the minimum regression suite and rerun it",
}

_CLOSE_GATE_BLOCKER_ACTIONS: dict[str, str] = {
    "performance baseline passed": "Rerun algorithm performance baseline and fix regressions.",
    "latency baseline passed": "Rerun latency benchmark and reduce latency regressions.",
    "rollback_baseline_not_tagged": (
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate."
    ),
    "rollback version marked": (
        "Run `make prepare-rollback-tag` to create a rollback tag for the release candidate."
    ),
    "minimum regression passed": "Fix the minimum regression suite and rerun it.",
    "canary recommendation is PROCEED_CANARY": (
        "Resolve canary blockers until recommendation becomes PROCEED_CANARY."
    ),
    "decision is APPROVE_CANARY": (
        "Resolve decision blockers until decision becomes APPROVE_CANARY."
    ),
    "online_offline_replay_status=FAIL": "Resolve online/offline replay mismatches before close.",
}


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_incomplete_tasks(
    checklist: dict[str, Any],
    *,
    performance_required: bool,
    latency_required: bool,
    manual_items: list[str] | None = None,
) -> list[str]:
    tasks: list[str] = []
    if checklist.get("kpi_snapshot_updated") is not True:
        tasks.append("KPI 快照更新")
    if checklist.get("experiment_ids_assigned") is not True:
        tasks.append("实验编号分配完成")
    if checklist.get("change_log_complete") is not True:
        tasks.append("变更记录完整")
    if checklist.get("rollback_version_marked") is not True:
        tasks.append("回滚版本已标记")
    if checklist.get("minimum_regression_passed") is not True:
        tasks.append("最小回归通过")
    if performance_required and checklist.get("performance_baseline_passed") is not True:
        tasks.append("性能基线达标")
    if latency_required and checklist.get("latency_baseline_passed") is not True:
        tasks.append("延迟基线达标")
    if checklist.get("consistency_check_completed") is not True:
        tasks.append("一致性检查完成")
    if checklist.get("anomalies_attributed") is not True:
        tasks.append("异常项已归因")
    tasks.extend(manual_items or MANUAL_CHECKLIST_ITEMS)
    return tasks


def remediation_for_canary_blockers(blockers: list[str]) -> list[str]:
    return _dedupe_keep_order(
        [_CANARY_BLOCKER_REMEDIATIONS.get(str(blocker).strip(), "") for blocker in blockers]
    )


def build_close_gate_action_items(
    *,
    signoff_status: str,
    auto_blockers: list[str],
    manual_missing: list[str],
    role_signoffs_missing: list[str],
    close_ready: bool,
) -> list[str]:
    action_items: list[str] = []
    if signoff_status == "AUTO_BLOCKED":
        mapped = [_CLOSE_GATE_BLOCKER_ACTIONS.get(item, "") for item in auto_blockers]
        action_items.extend(item for item in mapped if item)
        if not action_items:
            action_items.append("Resolve auto blockers to clear AUTO_BLOCKED status.")
    if manual_missing:
        action_items.append("Complete remaining manual checks.")
    if role_signoffs_missing:
        action_items.append("Collect all role sign-offs (Research/Engineering/Risk).")
    if not action_items and not close_ready:
        action_items.append("Review sign-off payload and set status to READY_FOR_CLOSE.")
    return _dedupe_keep_order(action_items)
