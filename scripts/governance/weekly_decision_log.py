#!/usr/bin/env python3
"""Generate structured weekly decision/rollback log from governance artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def _build_report(audit: dict[str, Any], canary: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("summary", {})
    checklist = audit.get("checklist", {})
    rollback = audit.get("rollback_marker", {})
    recommendation = canary.get("recommendation", "HOLD")
    blockers = list(canary.get("blockers", []))
    warnings = list(canary.get("warnings", []))

    decision = "APPROVE_CANARY" if recommendation == "PROCEED_CANARY" else "HOLD_AND_REMEDIATE"
    decision_reason = (
        "All automated gates passed." if decision == "APPROVE_CANARY" else "; ".join(blockers)
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decision_reason": decision_reason,
        "recommendation": recommendation,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": {
            "risk_exceptions": int(summary.get("exceptions", 0) or 0),
            "consistency_exceptions": int(summary.get("consistency_exceptions", 0) or 0),
            "minimum_regression_passed": bool(checklist.get("minimum_regression_passed")),
            "rollback_version_marked": bool(checklist.get("rollback_version_marked")),
            "rollback_marker_from_tag": bool(checklist.get("rollback_marker_from_tag")),
        },
        "rollback": {
            "reference": rollback.get("tag", ""),
            "source": rollback.get("source", ""),
        },
        "follow_up_tasks": list(audit.get("incomplete_tasks", [])),
    }


def _to_markdown(report: dict[str, Any]) -> str:
    ev = report["evidence"]
    rollback = report["rollback"]
    lines: list[str] = []
    lines.append("# Weekly Decision Log")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Decision: `{report['decision']}`")
    lines.append(f"- Recommendation: `{report['recommendation']}`")
    lines.append(f"- Reason: {report['decision_reason']}")
    lines.append("")
    if report["blockers"]:
        lines.append(f"- Blockers: `{'; '.join(report['blockers'])}`")
    else:
        lines.append("- Blockers: `_none_`")
    if report["warnings"]:
        lines.append(f"- Warnings: `{'; '.join(report['warnings'])}`")
    else:
        lines.append("- Warnings: `_none_`")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(f"- Risk exceptions: `{ev['risk_exceptions']}`")
    lines.append(f"- Consistency exceptions: `{ev['consistency_exceptions']}`")
    lines.append(f"- Minimum regression passed: `{ev['minimum_regression_passed']}`")
    lines.append(f"- Rollback baseline marked: `{ev['rollback_version_marked']}`")
    lines.append(f"- Rollback baseline from tag: `{ev['rollback_marker_from_tag']}`")
    lines.append("")
    lines.append("## Rollback Reference")
    lines.append("")
    lines.append(f"- Reference: `{rollback.get('reference') or 'TBD'}`")
    lines.append(f"- Source: `{rollback.get('source') or 'unknown'}`")
    lines.append("")
    lines.append("## Follow-up Tasks")
    lines.append("")
    tasks = report.get("follow_up_tasks", [])
    if tasks:
        for task in tasks:
            lines.append(f"- [ ] {task}")
    else:
        lines.append("- [x] None")
    lines.append("")
    lines.append("## Sign-off")
    lines.append("")
    lines.append("- Research: ")
    lines.append("- Engineering: ")
    lines.append("- Risk: ")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly decision/rollback log.")
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
        "--output-md",
        default="artifacts/weekly-decision-log.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/weekly-decision-log.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when decision is HOLD_AND_REMEDIATE.",
    )
    args = parser.parse_args()

    audit = _load_json(Path(args.audit_json).resolve())
    canary = _load_json(Path(args.canary_json).resolve())
    report = _build_report(audit, canary)

    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Weekly decision log: {report['decision']}.")
    if report["decision"] == "HOLD_AND_REMEDIATE" and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
