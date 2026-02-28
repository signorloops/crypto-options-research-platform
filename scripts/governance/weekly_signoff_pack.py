#!/usr/bin/env python3
"""Aggregate weekly governance outputs into a manual sign-off package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANUAL_ITEMS: list[tuple[str, str]] = [
    ("gray_release_completed", "灰度发布完成"),
    ("observation_24h_completed", "24h 观察完成"),
    ("rollback_decision_recorded", "是否触发回滚已决策"),
    ("pnl_attribution_confirmed", "收益归因表确认"),
    ("change_and_rollback_recorded", "变更与回滚记录"),
    ("adr_signed", "ADR"),
]

ROLE_SIGNOFF_ITEMS: list[tuple[str, str]] = [
    ("research", "Research 签字"),
    ("engineering", "Engineering 签字"),
    ("risk", "Risk 签字"),
]

TASK_TO_MANUAL_KEY: dict[str, str] = {
    "灰度发布完成": "gray_release_completed",
    "24h 观察完成": "observation_24h_completed",
    "是否触发回滚已决策": "rollback_decision_recorded",
    "收益归因表": "pnl_attribution_confirmed",
    "收益归因表确认": "pnl_attribution_confirmed",
    "变更与回滚记录": "change_and_rollback_recorded",
    "ADR": "adr_signed",
}

TASK_CANONICAL_LABEL: dict[str, str] = {
    "收益归因表": "收益归因表确认",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _normalize_manual_status(raw: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for key, _label in MANUAL_ITEMS:
        status[key] = _as_bool(raw.get(key))

    signoffs_raw = raw.get("signoffs")
    signoffs_map = signoffs_raw if isinstance(signoffs_raw, dict) else {}
    signoffs: dict[str, str] = {}
    for role, _label in ROLE_SIGNOFF_ITEMS:
        value = signoffs_map.get(role)
        signoffs[role] = str(value).strip() if value is not None else ""
    status["signoffs"] = signoffs
    return status


def _default_manual_status_template() -> dict[str, Any]:
    return _normalize_manual_status({})


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _build_report(
    *,
    audit: dict[str, Any],
    canary: dict[str, Any],
    decision: dict[str, Any],
    attribution: dict[str, Any],
    manual_status: dict[str, Any],
    consistency_replay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = _normalize_manual_status(manual_status)
    checklist = audit.get("checklist", {})
    summary = audit.get("summary", {})
    recommendation = str(canary.get("recommendation", "")).strip()
    decision_value = str(decision.get("decision", "")).strip()
    replay = consistency_replay if isinstance(consistency_replay, dict) else {}
    consistency_replay_status = str(replay.get("status", "")).strip().upper()

    auto_checks = [
        {
            "label": "risk exceptions = 0",
            "passed": int(summary.get("exceptions", 0) or 0) == 0,
        },
        {
            "label": "consistency exceptions = 0",
            "passed": int(summary.get("consistency_exceptions", 0) or 0) == 0,
        },
        {
            "label": "minimum regression passed",
            "passed": bool(checklist.get("minimum_regression_passed")),
        },
        {
            "label": "rollback version marked",
            "passed": bool(checklist.get("rollback_version_marked")),
        },
        {
            "label": "canary recommendation is PROCEED_CANARY",
            "passed": recommendation == "PROCEED_CANARY",
        },
        {
            "label": "decision is APPROVE_CANARY",
            "passed": decision_value == "APPROVE_CANARY",
        },
        {
            "label": "online/offline replay status != FAIL",
            "passed": consistency_replay_status != "FAIL",
        },
    ]

    auto_blockers = [item["label"] for item in auto_checks if not item["passed"]]
    if consistency_replay_status == "FAIL":
        auto_blockers.append("online_offline_replay_status=FAIL")

    manual_items: list[dict[str, Any]] = []
    pending_items: list[str] = []
    for key, label in MANUAL_ITEMS:
        done = bool(status.get(key))
        manual_items.append({"key": key, "label": label, "done": done})
        if not done:
            pending_items.append(label)

    role_signoffs: list[dict[str, Any]] = []
    for role, label in ROLE_SIGNOFF_ITEMS:
        signer = status["signoffs"].get(role, "")
        done = bool(signer)
        role_signoffs.append({"role": role, "label": label, "done": done, "value": signer})
        if not done:
            pending_items.append(label)

    for task in list(audit.get("incomplete_tasks", [])) + list(decision.get("follow_up_tasks", [])):
        task_text = str(task).strip()
        if not task_text:
            continue
        manual_key = TASK_TO_MANUAL_KEY.get(task_text)
        if manual_key and bool(status.get(manual_key)):
            continue
        pending_items.append(TASK_CANONICAL_LABEL.get(task_text, task_text))
    replay_pending = consistency_replay_status in {"", "PENDING_DATA"}
    if replay_pending:
        pending_items.append("线上/线下一致性回放数据待联调")

    pending_items = _dedupe_keep_order(pending_items)
    if auto_blockers:
        overall_status = "AUTO_BLOCKED"
    elif pending_items:
        overall_status = "PENDING_MANUAL_SIGNOFF"
    else:
        overall_status = "READY_FOR_CLOSE"

    attribution_rows = attribution.get("attribution_snapshot", [])
    attribution_count = len(attribution_rows) if isinstance(attribution_rows, list) else 0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "auto_blockers": auto_blockers,
        "pending_items": pending_items,
        "auto_checks": auto_checks,
        "manual_items": manual_items,
        "role_signoffs": role_signoffs,
        "inputs": {
            "canary_recommendation": recommendation,
            "decision": decision_value,
            "attribution_rows": attribution_count,
            "online_offline_replay_status": consistency_replay_status,
        },
        "summary": {
            "auto_blockers": len(auto_blockers),
            "pending_items": len(pending_items),
            "manual_done": sum(1 for item in manual_items if item["done"]),
            "manual_total": len(manual_items),
            "role_signoffs_done": sum(1 for item in role_signoffs if item["done"]),
            "role_signoffs_total": len(role_signoffs),
        },
        "manual_status": status,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Weekly Sign-off Pack")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(
        "- Inputs: "
        f"canary=`{report['inputs']['canary_recommendation'] or 'n/a'}`, "
        f"decision=`{report['inputs']['decision'] or 'n/a'}`, "
        f"attribution_rows=`{report['inputs']['attribution_rows']}`, "
        f"online_offline_replay=`{report['inputs']['online_offline_replay_status'] or 'n/a'}`"
    )
    lines.append("")
    lines.append("## Auto Checks")
    lines.append("")
    for item in report["auto_checks"]:
        mark = "[x]" if item["passed"] else "[ ]"
        lines.append(f"- {mark} {item['label']}")
    lines.append("")
    lines.append("## Manual Checks")
    lines.append("")
    for item in report["manual_items"]:
        mark = "[x]" if item["done"] else "[ ]"
        lines.append(f"- {mark} {item['label']}")
    lines.append("")
    lines.append("## Role Sign-offs")
    lines.append("")
    for item in report["role_signoffs"]:
        mark = "[x]" if item["done"] else "[ ]"
        value = item["value"] if item["value"] else "TBD"
        lines.append(f"- {mark} {item['label']}: `{value}`")
    lines.append("")
    lines.append("## Pending Items")
    lines.append("")
    if report["pending_items"]:
        for item in report["pending_items"]:
            lines.append(f"- [ ] {item}")
    else:
        lines.append("- [x] None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly sign-off package.")
    parser.add_argument(
        "--audit-json",
        default="artifacts/weekly-operating-audit.json",
        help="Path to weekly audit JSON.",
    )
    parser.add_argument(
        "--canary-json",
        default="artifacts/weekly-canary-checklist.json",
        help="Path to weekly canary checklist JSON.",
    )
    parser.add_argument(
        "--decision-json",
        default="artifacts/weekly-decision-log.json",
        help="Path to weekly decision log JSON.",
    )
    parser.add_argument(
        "--attribution-json",
        default="artifacts/weekly-pnl-attribution.json",
        help="Path to weekly attribution JSON.",
    )
    parser.add_argument(
        "--manual-status-json",
        default="artifacts/weekly-manual-status.json",
        help="Path to manual confirmation JSON. Defaults to all-unchecked when file is missing.",
    )
    parser.add_argument(
        "--consistency-replay-json",
        default="artifacts/online-offline-consistency-replay.json",
        help="Path to online/offline consistency replay JSON.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/weekly-signoff-pack.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/weekly-signoff-pack.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless status is READY_FOR_CLOSE.",
    )
    args = parser.parse_args()

    audit = _load_json(Path(args.audit_json).resolve())
    canary = _load_json(Path(args.canary_json).resolve())
    decision = _load_json(Path(args.decision_json).resolve())
    attribution = _load_json(Path(args.attribution_json).resolve())
    manual_status_path = Path(args.manual_status_json).resolve()
    consistency_replay = _load_optional_json(Path(args.consistency_replay_json).resolve())
    if not manual_status_path.exists():
        manual_status_path.parent.mkdir(parents=True, exist_ok=True)
        manual_status_path.write_text(
            json.dumps(_default_manual_status_template(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    manual_status = _load_optional_json(manual_status_path)

    report = _build_report(
        audit=audit,
        canary=canary,
        decision=decision,
        attribution=attribution,
        manual_status=manual_status,
        consistency_replay=consistency_replay,
    )

    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Weekly sign-off pack: {report['status']}.")
    if args.strict and report["status"] != "READY_FOR_CLOSE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
