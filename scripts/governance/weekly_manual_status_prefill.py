#!/usr/bin/env python3
"""Auto-prefill objective weekly manual-status fields from generated artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.report_utils import (
    as_bool as _as_bool,
    load_json_object as _load_json,
    load_optional_json_object as _load_optional_json,
    write_json as _write_json,
    write_markdown as _write_markdown,
)

MANUAL_KEYS: list[str] = [
    "gray_release_completed",
    "observation_24h_completed",
    "rollback_decision_recorded",
    "pnl_attribution_confirmed",
    "change_and_rollback_recorded",
    "adr_signed",
]

ROLE_KEYS: list[str] = ["research", "engineering", "risk"]
MANUAL_LABELS: list[tuple[str, str]] = [
    ("gray_release_completed", "灰度发布完成"),
    ("observation_24h_completed", "24h 观察完成"),
    ("rollback_decision_recorded", "是否触发回滚已决策"),
    ("pnl_attribution_confirmed", "收益归因表确认"),
    ("change_and_rollback_recorded", "变更与回滚记录"),
    ("adr_signed", "ADR"),
]
ROLE_LABELS: list[tuple[str, str]] = [
    ("research", "Research 签字"),
    ("engineering", "Engineering 签字"),
    ("risk", "Risk 签字"),
]


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_status(raw: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for key in MANUAL_KEYS:
        status[key] = _as_bool(raw.get(key))

    signoffs_raw = raw.get("signoffs")
    signoffs_map = signoffs_raw if isinstance(signoffs_raw, dict) else {}
    signoffs: dict[str, str] = {}
    for role in ROLE_KEYS:
        value = signoffs_map.get(role)
        signoffs[role] = str(value).strip() if value is not None else ""
    status["signoffs"] = signoffs
    return status


def _derive_autofill(decision: dict[str, Any], attribution: dict[str, Any]) -> dict[str, bool]:
    decision_text = str(decision.get("decision", "")).strip()
    rollback_raw = decision.get("rollback")
    rollback_map = rollback_raw if isinstance(rollback_raw, dict) else {}
    rollback_ref = str(rollback_map.get("reference", "")).strip()

    summary_raw = attribution.get("summary")
    summary_map = summary_raw if isinstance(summary_raw, dict) else {}
    strategies = _as_int(summary_map.get("strategies")) or 0
    missing_entries = _as_int(summary_map.get("missing_entries"))
    attribution_complete = strategies > 0 and missing_entries == 0

    return {
        "rollback_decision_recorded": bool(decision_text),
        "change_and_rollback_recorded": bool(decision_text and rollback_ref),
        "pnl_attribution_confirmed": attribution_complete,
    }


def _apply_prefill(
    status: dict[str, Any], prefill: dict[str, bool]
) -> tuple[dict[str, Any], list[str]]:
    updated = _normalize_status(status)
    changed_keys: list[str] = []
    for key, flag in prefill.items():
        if not flag:
            continue
        if not updated.get(key):
            updated[key] = True
            changed_keys.append(key)
    return updated, changed_keys


def _to_markdown(
    *,
    status: dict[str, Any],
    decision: dict[str, Any],
    changed_keys: list[str],
) -> str:
    decision_text = str(decision.get("decision", "")).strip() or "TBD"
    rollback_raw = decision.get("rollback")
    rollback_map = rollback_raw if isinstance(rollback_raw, dict) else {}
    rollback_ref = str(rollback_map.get("reference", "")).strip() or "TBD"
    rollback_source = str(rollback_map.get("source", "")).strip() or "unknown"
    changed_text = ", ".join(changed_keys) if changed_keys else "none"

    lines: list[str] = []
    lines.append("# Weekly Manual Status")
    lines.append("")
    lines.append(f"- Decision: `{decision_text}`")
    lines.append(f"- Rollback reference: `{rollback_ref}`")
    lines.append(f"- Rollback source: `{rollback_source}`")
    lines.append(f"- Auto-filled keys: `{changed_text}`")
    lines.append("")
    lines.append("## Manual Checks")
    lines.append("")
    for key, label in MANUAL_LABELS:
        mark = "[x]" if bool(status.get(key)) else "[ ]"
        lines.append(f"- {mark} {label}")
    lines.append("")
    lines.append("## Role Sign-offs")
    lines.append("")
    signoffs = status.get("signoffs", {})
    signoff_map = signoffs if isinstance(signoffs, dict) else {}
    for role, label in ROLE_LABELS:
        signer = str(signoff_map.get(role, "")).strip() or "TBD"
        mark = "[x]" if signer != "TBD" else "[ ]"
        lines.append(f"- {mark} {label}: `{signer}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-prefill objective weekly manual-status fields."
    )
    parser.add_argument(
        "--decision-json",
        default="artifacts/weekly-decision-log.json",
        help="Path to weekly decision log JSON.",
    )
    parser.add_argument(
        "--attribution-json",
        default="artifacts/weekly-pnl-attribution.json",
        help="Path to weekly PnL attribution JSON.",
    )
    parser.add_argument(
        "--manual-status-json",
        default="artifacts/weekly-manual-status.json",
        help="Path to manual status JSON (also used as default output path).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path. Defaults to --manual-status-json.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional markdown checklist path. Disabled when empty.",
    )
    args = parser.parse_args()

    decision = _load_optional_json(Path(args.decision_json).resolve())
    attribution = _load_optional_json(Path(args.attribution_json).resolve())

    manual_status_path = Path(args.manual_status_json).resolve()
    output_path = Path(args.output_json).resolve() if args.output_json else manual_status_path

    existing_status = _load_optional_json(manual_status_path)
    status = _normalize_status(existing_status)
    prefill = _derive_autofill(decision, attribution)
    updated_status, changed_keys = _apply_prefill(status, prefill)

    _write_json(output_path, updated_status)
    if args.output_md:
        _write_markdown(
            Path(args.output_md).resolve(),
            _to_markdown(status=updated_status, decision=decision, changed_keys=changed_keys),
        )

    changed_text = ", ".join(changed_keys) if changed_keys else "none"
    print("Weekly manual status prefill: " f"updated_keys={changed_text}, output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
