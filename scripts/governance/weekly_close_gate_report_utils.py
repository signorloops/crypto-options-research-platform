"""Helper functions for weekly close-gate report evaluation and rendering."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.governance.report_utils import (
    load_json_object as _load_json,
    write_json as _write_json,
    write_markdown as _write_markdown,
)
from scripts.governance.status_action_utils import build_close_gate_action_items
from scripts.governance.weekly_close_gate_utils import (
    build_close_gate_summary,
    collect_open_labels,
)
from scripts.governance.weekly_operating_render_utils import (
    build_close_gate_markdown,
    build_close_gate_pr_brief,
)


def _to_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = str(entry).strip()
        if text:
            items.append(text)
    return items


def load_close_gate_snapshot(signoff_json_path: Path) -> tuple[dict[str, Any], str]:
    if not signoff_json_path.exists():
        return {}, "missing_signoff_json"
    try:
        payload = _load_json(signoff_json_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, "invalid_signoff_json"
    return payload, ""


def evaluate_close_gate(signoff_json_path: Path) -> tuple[bool, str, dict[str, Any]]:
    payload, load_error = load_close_gate_snapshot(signoff_json_path)
    if load_error:
        return False, load_error, {}
    status = str(payload.get("status", "")).strip().upper()
    if status == "READY_FOR_CLOSE":
        return True, status, payload
    if status:
        return False, f"status={status}", payload
    return False, "status=UNKNOWN", payload


def build_close_gate_report(
    *,
    signoff_json_path: Path,
    close_ready: bool,
    close_detail: str,
    signoff_payload: dict[str, Any],
) -> dict[str, Any]:
    manual_missing = collect_open_labels(signoff_payload.get("manual_items"))
    role_signoffs_missing = collect_open_labels(signoff_payload.get("role_signoffs"))
    signoff_status = str(signoff_payload.get("status", "")).strip().upper()
    auto_blockers = _to_text_list(signoff_payload.get("auto_blockers"))
    pending_items = _to_text_list(signoff_payload.get("pending_items"))
    action_items = build_close_gate_action_items(
        signoff_status=signoff_status,
        auto_blockers=auto_blockers,
        manual_missing=manual_missing,
        role_signoffs_missing=role_signoffs_missing,
        close_ready=close_ready,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if close_ready else "FAIL",
        "gate": "READY_FOR_CLOSE",
        "reason": close_detail,
        "signoff_json": str(signoff_json_path),
        "signoff_status": signoff_status or "UNKNOWN",
        "auto_blockers": auto_blockers,
        "pending_items": pending_items,
        "manual_missing": manual_missing,
        "role_signoffs_missing": role_signoffs_missing,
        "action_items": action_items,
        "pr_brief": build_close_gate_pr_brief(
            close_ready=close_ready,
            close_detail=close_detail,
            signoff_status=signoff_status,
            auto_blockers=auto_blockers,
            pending_items=pending_items,
            action_items=action_items,
        ),
        "summary": build_close_gate_summary(
            auto_blockers=auto_blockers,
            pending_items=pending_items,
            manual_missing=manual_missing,
            role_signoffs_missing=role_signoffs_missing,
        ),
    }


def write_close_gate_report(
    *,
    signoff_json_path: Path,
    close_gate_md: Path,
    close_gate_json: Path,
    close_ready: bool,
    close_detail: str,
    signoff_payload: dict[str, Any],
) -> None:
    close_report = build_close_gate_report(
        signoff_json_path=signoff_json_path,
        close_ready=close_ready,
        close_detail=close_detail,
        signoff_payload=signoff_payload,
    )
    _write_markdown(close_gate_md, build_close_gate_markdown(close_report))
    _write_json(close_gate_json, close_report)
