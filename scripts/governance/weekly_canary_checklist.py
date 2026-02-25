#!/usr/bin/env python3
"""Generate canary rollout + 24h observation checklist from weekly governance artifacts."""

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


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "N/A"


def _build_report(audit: dict[str, Any], attribution: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("summary", {})
    checklist = audit.get("checklist", {})
    thresholds = audit.get("thresholds", {})
    regression = audit.get("regression", {})
    rollback_marker = audit.get("rollback_marker", {})
    consistency_exceptions = int(summary.get("consistency_exceptions", 0) or 0)
    risk_exceptions = int(summary.get("exceptions", 0) or 0)
    regression_ok = bool(regression.get("passed"))

    blockers: list[str] = []
    if risk_exceptions > 0:
        blockers.append(f"risk_exceptions={risk_exceptions}")
    if consistency_exceptions > 0:
        blockers.append(f"consistency_exceptions={consistency_exceptions}")
    if not regression_ok:
        blockers.append("minimum_regression_failed")
    if checklist.get("rollback_version_marked") is not True:
        blockers.append("rollback_baseline_not_tagged")

    recommendation = "HOLD" if blockers else "PROCEED_CANARY"
    strategy_rows = audit.get("kpi_snapshot", [])
    attribution_rows = attribution.get("attribution_snapshot", [])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "blockers": blockers,
        "preconditions": {
            "risk_exceptions": risk_exceptions,
            "consistency_exceptions": consistency_exceptions,
            "minimum_regression_passed": regression_ok,
            "rollback_version_marked": checklist.get("rollback_version_marked"),
        },
        "thresholds": thresholds,
        "rollback_marker": rollback_marker,
        "kpi_snapshot": strategy_rows,
        "attribution_snapshot": attribution_rows,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    pre = report["preconditions"]
    thresholds = report.get("thresholds", {})
    rollback = report.get("rollback_marker", {})
    lines: list[str] = []
    lines.append("# Weekly Canary Checklist")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Recommendation: `{report['recommendation']}`")
    if report["blockers"]:
        lines.append(f"- Blockers: `{'; '.join(report['blockers'])}`")
    else:
        lines.append("- Blockers: `_none_`")
    lines.append("")
    lines.append("## Preconditions (Auto)")
    lines.append("")
    lines.append(f"- Risk exceptions: `{pre['risk_exceptions']}`")
    lines.append(f"- Consistency exceptions: `{pre['consistency_exceptions']}`")
    lines.append(f"- Minimum regression passed: `{_fmt_bool(pre['minimum_regression_passed'])}`")
    lines.append(f"- Rollback baseline marked: `{_fmt_bool(pre['rollback_version_marked'])}`")
    lines.append("")
    lines.append("## Rollback Baseline")
    lines.append("")
    lines.append(f"- Reference: `{rollback.get('tag') or 'TBD'}`")
    lines.append("")
    lines.append("## Canary Plan (Manual Fill)")
    lines.append("")
    lines.append("- [ ] Step 1: 5% exposure / small inventory cap")
    lines.append("- [ ] Step 2: 20% exposure if no alerts in first 2h")
    lines.append("- [ ] Step 3: 50% exposure after 24h pass")
    lines.append("")
    lines.append("## 24h Observation Checklist")
    lines.append("")
    lines.append("- [ ] PnL / Sharpe within expected range")
    lines.append("- [ ] Max drawdown <= threshold")
    lines.append("- [ ] VaR breach rate <= threshold")
    lines.append("- [ ] Fill calibration error <= threshold")
    lines.append("- [ ] No abnormal consistency drift")
    lines.append("")
    lines.append("## Triggered Rollback Conditions")
    lines.append("")
    lines.append(f"- Sharpe < `{thresholds.get('min_sharpe', 'n/a')}`")
    lines.append(f"- |Max Drawdown| > `{thresholds.get('max_abs_drawdown', 'n/a')}`")
    lines.append(f"- VaR breach rate > `{thresholds.get('max_var_breach_rate', 'n/a')}`")
    lines.append(
        f"- Fill calibration error > `{thresholds.get('max_fill_calibration_error', 'n/a')}`"
    )
    lines.append("")
    lines.append("## Sign-off")
    lines.append("")
    lines.append("- Research: ")
    lines.append("- Engineering: ")
    lines.append("- Risk: ")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly canary rollout checklist.")
    parser.add_argument(
        "--audit-json",
        default="artifacts/weekly-operating-audit.json",
        help="Path to weekly operating audit JSON.",
    )
    parser.add_argument(
        "--attribution-json",
        default="artifacts/weekly-pnl-attribution.json",
        help="Path to weekly PnL attribution JSON.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/weekly-canary-checklist.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/weekly-canary-checklist.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when recommendation is HOLD.",
    )
    args = parser.parse_args()

    audit = _load_json(Path(args.audit_json).resolve())
    attribution = _load_json(Path(args.attribution_json).resolve())
    report = _build_report(audit, attribution)

    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if report["recommendation"] == "HOLD":
        print(f"Weekly canary checklist: HOLD ({'; '.join(report['blockers'])}).")
        return 2 if args.strict else 0
    print("Weekly canary checklist: PROCEED_CANARY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
