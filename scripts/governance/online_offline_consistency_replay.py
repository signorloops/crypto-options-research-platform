#!/usr/bin/env python3
"""Generate online/offline consistency replay report from governance artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_offline_consistency_exceptions": 0,
    "max_online_alerts": 0,
    "max_online_abs_deviation_bps": 300.0,
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


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _load_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_THRESHOLDS)
    raw = _load_json(path)
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key, default in DEFAULT_THRESHOLDS.items():
        value = _as_float(raw.get(key))
        thresholds[key] = default if value is None else float(value)
    return thresholds


def _build_report(
    *,
    audit: dict[str, Any],
    live: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    audit_summary_raw = audit.get("summary")
    audit_summary = audit_summary_raw if isinstance(audit_summary_raw, dict) else {}
    live_summary_raw = live.get("summary")
    live_summary = live_summary_raw if isinstance(live_summary_raw, dict) else {}
    alerts_raw = live.get("alerts")
    alerts = alerts_raw if isinstance(alerts_raw, list) else []

    offline_consistency_exceptions = _as_int(audit_summary.get("consistency_exceptions")) or 0
    offline_consistency_pairs = _as_int(audit_summary.get("consistency_pairs")) or 0

    online_rows = _as_int(live_summary.get("n_rows")) or 0
    online_alerts = _as_int(live_summary.get("n_alerts")) or 0
    online_max_abs_deviation_bps = _as_float(live_summary.get("max_abs_deviation_bps"))
    online_mean_abs_deviation_bps = _as_float(live_summary.get("mean_abs_deviation_bps"))

    data_ready = online_rows > 0 and online_max_abs_deviation_bps is not None

    breaches: list[str] = []
    if offline_consistency_exceptions > int(thresholds["max_offline_consistency_exceptions"]):
        breaches.append(
            "offline_consistency_exceptions>"
            f"{int(thresholds['max_offline_consistency_exceptions'])}"
        )

    if data_ready:
        if online_alerts > int(thresholds["max_online_alerts"]):
            breaches.append(f"online_alerts>{int(thresholds['max_online_alerts'])}")
        if (
            online_max_abs_deviation_bps is not None
            and online_max_abs_deviation_bps > thresholds["max_online_abs_deviation_bps"]
        ):
            breaches.append(
                "online_max_abs_deviation_bps>"
                f"{thresholds['max_online_abs_deviation_bps']}"
            )

    if breaches:
        status = "FAIL"
    elif not data_ready:
        status = "PENDING_DATA"
    else:
        status = "PASS"

    top_alerts: list[dict[str, Any]] = []
    root_cause_candidates: list[str] = []
    for item in alerts[:5]:
        if not isinstance(item, dict):
            continue
        venue = str(item.get("venue", "")).strip() or "unknown"
        maturity = item.get("maturity")
        delta = item.get("delta")
        deviation = _as_float(item.get("deviation_bps"))
        top_alerts.append(
            {
                "venue": venue,
                "maturity": maturity,
                "delta": delta,
                "deviation_bps": deviation,
            }
        )
        if deviation is None:
            root_cause_candidates.append(f"{venue}: high deviation alert without numeric bps")
        else:
            root_cause_candidates.append(
                f"{venue}: deviation_bps={deviation:.2f}, maturity={maturity}, delta={delta}"
            )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "breaches": breaches,
        "thresholds": thresholds,
        "summary": {
            "data_ready": data_ready,
            "offline_consistency_pairs": offline_consistency_pairs,
            "offline_consistency_exceptions": offline_consistency_exceptions,
            "online_rows": online_rows,
            "online_alerts": online_alerts,
            "online_max_abs_deviation_bps": online_max_abs_deviation_bps,
            "online_mean_abs_deviation_bps": online_mean_abs_deviation_bps,
        },
        "top_alerts": top_alerts,
        "root_cause_candidates": root_cause_candidates,
        "inputs": {
            "audit_generated_at_utc": audit.get("generated_at_utc", ""),
            "live_generated_at_utc": live.get("generated_at_utc", ""),
        },
    }


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep, *body])


def _to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Online/Offline Consistency Replay")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Data ready: `{summary['data_ready']}`")
    lines.append("")
    lines.append("## Online vs Offline Delta Summary")
    lines.append("")
    lines.append(
        f"- Offline consistency exceptions: `{summary['offline_consistency_exceptions']}` "
        f"(pairs=`{summary['offline_consistency_pairs']}`)"
    )
    lines.append(f"- Online rows: `{summary['online_rows']}`")
    lines.append(f"- Online alerts: `{summary['online_alerts']}`")
    lines.append(
        "- Online max abs deviation (bps): "
        f"`{_fmt_float(summary['online_max_abs_deviation_bps'])}`"
    )
    lines.append(
        "- Online mean abs deviation (bps): "
        f"`{_fmt_float(summary['online_mean_abs_deviation_bps'])}`"
    )
    lines.append("")
    lines.append("## Threshold Breaches")
    lines.append("")
    if report["breaches"]:
        for breach in report["breaches"]:
            lines.append(f"- [ ] {breach}")
    else:
        lines.append("- [x] None")
    lines.append("")
    lines.append("## Top Alerts")
    lines.append("")
    lines.append(_format_table(report["top_alerts"], ["venue", "maturity", "delta", "deviation_bps"]))
    lines.append("")
    lines.append("## Root-Cause Candidates")
    lines.append("")
    if report["root_cause_candidates"]:
        for item in report["root_cause_candidates"]:
            lines.append(f"- {item}")
    else:
        lines.append("_none_")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate online/offline consistency replay report.")
    parser.add_argument(
        "--audit-json",
        default="artifacts/weekly-operating-audit.json",
        help="Path to weekly operating audit JSON.",
    )
    parser.add_argument(
        "--live-json",
        default="artifacts/live-deviation-snapshot.json",
        help="Path to live deviation snapshot JSON.",
    )
    parser.add_argument(
        "--thresholds",
        default="config/online_offline_consistency_thresholds.json",
        help="Path to online/offline consistency thresholds JSON.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/online-offline-consistency-replay.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/online-offline-consistency-replay.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless status is PASS.",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    audit = _load_json((repo_root / args.audit_json).resolve())
    live = _load_optional_json((repo_root / args.live_json).resolve())
    thresholds = _load_thresholds((repo_root / args.thresholds).resolve())

    report = _build_report(audit=audit, live=live, thresholds=thresholds)

    output_md = (repo_root / args.output_md).resolve()
    output_json = (repo_root / args.output_json).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_to_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Online/offline consistency replay: {report['status']}.")
    if args.strict and report["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
