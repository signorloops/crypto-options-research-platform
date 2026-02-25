#!/usr/bin/env python3
"""Generate an ADR draft from weekly operating audit output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid audit report JSON: {path}")
    return data


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "N/A"


def _build_markdown(report: dict[str, Any], owner: str) -> str:
    summary = report.get("summary", {})
    checklist = report.get("checklist", {})
    regression = report.get("regression", {})
    rollback = report.get("rollback_marker", {})
    change_log = report.get("change_log", {})
    risk_exceptions = report.get("risk_exceptions", [])
    incomplete = report.get("incomplete_tasks", [])

    decision = (
        "Proceed with staged rollout"
        if summary.get("exceptions", 0) == 0 and checklist.get("minimum_regression_passed") is True
        else "Hold rollout and remediate risk findings"
    )

    lines: list[str] = []
    lines.append("# Weekly ADR Draft")
    lines.append("")
    lines.append(f"- Owner: `{owner}`")
    lines.append(f"- Generated from audit: `{report.get('generated_at_utc', 'unknown')}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- {decision}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(f"- Strategies covered: `{summary.get('strategies', 0)}`")
    lines.append(f"- Risk exceptions: `{summary.get('exceptions', 0)}`")
    lines.append(f"- Regression status: `{_fmt_bool(regression.get('passed'))}`")
    lines.append(f"- Change log status: `{_fmt_bool(checklist.get('change_log_complete'))}`")
    lines.append(
        f"- Rollback baseline status: `{_fmt_bool(checklist.get('rollback_version_marked'))}`"
    )
    lines.append("")
    lines.append("## Rollback Baseline")
    lines.append("")
    rollback_ref = rollback.get("tag") or "TBD"
    lines.append(f"- Reference: `{rollback_ref}`")
    lines.append("")
    lines.append("## Risk Findings")
    lines.append("")
    if risk_exceptions:
        for finding in risk_exceptions:
            strategy = finding.get("strategy", "unknown")
            detail = finding.get("breached_rules", "")
            lines.append(f"- `{strategy}`: {detail}")
    else:
        lines.append("- No threshold violations in this audit window.")
    lines.append("")
    lines.append("## Recent Changes")
    lines.append("")
    entries = change_log.get("entries", [])
    if entries:
        for entry in entries[:10]:
            lines.append(
                f"- {entry.get('date', '')} `{entry.get('commit', '')}` {entry.get('subject', '')}"
            )
    else:
        lines.append("- No change log entries captured.")
    lines.append("")
    lines.append("## Follow-up Actions")
    lines.append("")
    if incomplete:
        for task in incomplete:
            lines.append(f"- [ ] {task}")
    else:
        lines.append("- [x] No outstanding follow-up tasks.")
    lines.append("")
    lines.append("## Sign-off")
    lines.append("")
    lines.append("- Research: ")
    lines.append("- Engineering: ")
    lines.append("- Risk: ")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly ADR draft from audit report.")
    parser.add_argument(
        "--audit-json",
        default="artifacts/weekly-operating-audit.json",
        help="Path to weekly operating audit JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/weekly-adr-draft.md",
        help="Path for generated ADR markdown draft.",
    )
    parser.add_argument(
        "--owner",
        default="TBD",
        help="Owner label embedded in ADR draft.",
    )
    args = parser.parse_args()

    audit_path = Path(args.audit_json).resolve()
    output_path = Path(args.output_md).resolve()

    report = _load_report(audit_path)
    markdown = _build_markdown(report, owner=args.owner)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Generated ADR draft: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
