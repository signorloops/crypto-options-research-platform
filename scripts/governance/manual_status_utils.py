#!/usr/bin/env python3
"""Shared helpers for weekly manual status normalization and updates."""

from __future__ import annotations

import shlex
from typing import Any

from scripts.governance.report_utils import as_bool as _as_bool

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

MANUAL_KEYS: list[str] = [key for key, _label in MANUAL_ITEMS]
ROLE_KEYS: list[str] = [role for role, _label in ROLE_SIGNOFF_ITEMS]

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

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}
_ROLE_PLACEHOLDERS = {
    "research": "research_owner",
    "engineering": "engineering_owner",
    "risk": "risk_owner",
}


def normalize_manual_status(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    status: dict[str, Any] = {key: _as_bool(payload.get(key)) for key in MANUAL_KEYS}
    signoffs_raw = payload.get("signoffs")
    signoffs_map = signoffs_raw if isinstance(signoffs_raw, dict) else {}
    status["signoffs"] = {
        role: str(signoffs_map.get(role, "")).strip() if signoffs_map.get(role) is not None else ""
        for role in ROLE_KEYS
    }
    return status


def default_manual_status_template() -> dict[str, Any]:
    return normalize_manual_status({})


def parse_bool_text(raw: str) -> bool:
    text = str(raw).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported boolean value: {raw}")


def apply_manual_status_updates(
    status: dict[str, Any] | None,
    *,
    manual_updates: dict[str, bool] | None = None,
    signoff_updates: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    updated = normalize_manual_status(status)
    changed_keys: list[str] = []

    for key, flag in (manual_updates or {}).items():
        if key not in MANUAL_KEYS:
            raise KeyError(key)
        normalized_flag = bool(flag)
        if updated.get(key) != normalized_flag:
            updated[key] = normalized_flag
            changed_keys.append(key)

    signoffs = dict(updated["signoffs"])
    for role, signer in (signoff_updates or {}).items():
        if role not in ROLE_KEYS:
            raise KeyError(role)
        normalized_signer = str(signer).strip()
        if signoffs.get(role, "") != normalized_signer:
            signoffs[role] = normalized_signer
            changed_keys.append(f"signoffs.{role}")
    updated["signoffs"] = signoffs
    return updated, changed_keys


def build_manual_status_markdown(
    *,
    status: dict[str, Any],
    decision: dict[str, Any] | None = None,
    changed_keys: list[str] | None = None,
) -> str:
    decision_payload = decision if isinstance(decision, dict) else {}
    rollback_raw = decision_payload.get("rollback")
    rollback_map = rollback_raw if isinstance(rollback_raw, dict) else {}
    decision_text = str(decision_payload.get("decision", "")).strip() or "TBD"
    rollback_ref = str(rollback_map.get("reference", "")).strip() or "TBD"
    rollback_source = str(rollback_map.get("source", "")).strip() or "unknown"
    changed_text = ", ".join(changed_keys or []) if changed_keys else "none"

    lines: list[str] = [
        "# Weekly Manual Status",
        "",
        f"- Decision: `{decision_text}`",
        f"- Rollback reference: `{rollback_ref}`",
        f"- Rollback source: `{rollback_source}`",
        f"- Updated keys: `{changed_text}`",
        "",
        "## Manual Checks",
        "",
    ]
    for key, label in MANUAL_ITEMS:
        mark = "[x]" if bool(status.get(key)) else "[ ]"
        lines.append(f"- {mark} {label}")

    lines.extend(["", "## Role Sign-offs", ""])
    signoff_map = status.get("signoffs", {})
    signoffs = signoff_map if isinstance(signoff_map, dict) else {}
    for role, label in ROLE_SIGNOFF_ITEMS:
        signer = str(signoffs.get(role, "")).strip() or "TBD"
        mark = "[x]" if signer != "TBD" else "[ ]"
        lines.append(f"- {mark} {label}: `{signer}`")
    lines.append("")
    return "\n".join(lines)


def build_manual_update_plan(status: dict[str, Any] | None) -> dict[str, Any]:
    updated = normalize_manual_status(status)
    signoffs = updated["signoffs"] if isinstance(updated.get("signoffs"), dict) else {}

    updater_pending_items: list[str] = []
    args: list[str] = []
    for key, label in MANUAL_ITEMS:
        if bool(updated.get(key)):
            continue
        updater_pending_items.append(label)
        args.extend(["--check", f"{key}=true"])

    for role, label in ROLE_SIGNOFF_ITEMS:
        signer = str(signoffs.get(role, "")).strip()
        if signer:
            continue
        updater_pending_items.append(label)
        args.extend(["--signoff", f"{role}={_ROLE_PLACEHOLDERS[role]}"])

    args_text = " ".join(shlex.quote(arg) for arg in args)
    command = (
        f"make weekly-manual-update MANUAL_ARGS={shlex.quote(args_text)}"
        if args_text
        else ""
    )
    return {
        "pending_items": updater_pending_items,
        "args": args,
        "suggested_command": command,
    }
