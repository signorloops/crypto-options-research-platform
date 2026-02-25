#!/usr/bin/env python3
"""Generate weekly PnL attribution table from backtest result files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRIC_KEYS = {
    "spread_capture": [
        "spread_capture",
        "total_spread_captured",
        "spread_pnl",
        "spread_revenue",
    ],
    "adverse_selection_cost": [
        "adverse_selection_cost",
        "adverse_cost",
        "toxicity_cost",
    ],
    "inventory_cost": [
        "inventory_cost",
        "inventory_holding_cost",
        "inventory_pnl_cost",
    ],
    "hedging_cost": [
        "hedging_cost",
        "hedge_cost",
        "delta_hedging_cost",
        "gamma_hedging_cost",
    ],
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return data


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num or num in (float("inf"), float("-inf")):
        return None
    return num


def _pick_first_numeric(candidates: list[dict[str, Any]], keys: list[str]) -> float | None:
    for mapping in candidates:
        for key in keys:
            if key in mapping:
                val = _to_float(mapping.get(key))
                if val is not None:
                    return val
    return None


def _pick_first_text(candidates: list[dict[str, Any]], keys: list[str]) -> str | None:
    for mapping in candidates:
        for key in keys:
            val = mapping.get(key)
            if val is None:
                continue
            text = str(val).strip()
            if text:
                return text
    return None


def _infer_experiment_id(source: Path) -> str:
    return f"AUTO-{source.stem.strip().replace(' ', '_')}"


def _extract_rows(raw: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows = []
    for strategy, payload in raw.items():
        if not isinstance(payload, dict):
            continue

        summary = payload.get("summary")
        metrics = payload.get("metrics")
        risk = payload.get("risk")
        summary_map = summary if isinstance(summary, dict) else {}
        metrics_map = metrics if isinstance(metrics, dict) else {}
        risk_map = risk if isinstance(risk, dict) else {}
        candidates = [summary_map, metrics_map, risk_map, payload]

        experiment_id = _pick_first_text(
            candidates, ["experiment_id", "experiment", "exp_id"]
        ) or _infer_experiment_id(source)

        extracted = {}
        missing_fields = []
        for metric_name, metric_keys in METRIC_KEYS.items():
            value = _pick_first_numeric(candidates, metric_keys)
            extracted[metric_name] = value
            if value is None:
                missing_fields.append(metric_name)

        rows.append(
            {
                "strategy": str(strategy),
                "source_file": source.name,
                "experiment_id": experiment_id,
                **extracted,
                "missing_fields": missing_fields,
            }
        )
    return rows


def _discover_input_files(results_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        [p for p in results_dir.rglob(pattern) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep, *body])


def _build_report(input_files: list[Path]) -> dict[str, Any]:
    latest_by_strategy: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, str]] = []

    for path in input_files:
        try:
            raw = _load_json(path)
            rows = _extract_rows(raw, path)
        except Exception as exc:  # pragma: no cover - defensive parser boundary
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue

        for row in rows:
            strategy = row["strategy"]
            if strategy not in latest_by_strategy:
                latest_by_strategy[strategy] = row

    snapshot = sorted(latest_by_strategy.values(), key=lambda r: r["strategy"])
    missing_entries = [
        {
            "strategy": row["strategy"],
            "source_file": row["source_file"],
            "missing_fields": ", ".join(row["missing_fields"]),
        }
        for row in snapshot
        if row["missing_fields"]
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in input_files],
        "summary": {
            "strategies": len(snapshot),
            "fully_populated": len(snapshot) - len(missing_entries),
            "missing_entries": len(missing_entries),
            "parse_errors": len(parse_errors),
        },
        "attribution_snapshot": snapshot,
        "missing_entries": missing_entries,
        "parse_errors": parse_errors,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Weekly PnL Attribution")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Input files: `{len(report['inputs'])}`")
    lines.append(f"- Strategies in snapshot: `{report['summary']['strategies']}`")
    lines.append(f"- Fully populated rows: `{report['summary']['fully_populated']}`")
    lines.append(f"- Rows with missing fields: `{report['summary']['missing_entries']}`")
    lines.append("")
    lines.append("## Attribution Snapshot")
    lines.append("")
    rows = []
    for row in report["attribution_snapshot"]:
        rows.append(
            {
                "strategy": row["strategy"],
                "source_file": row["source_file"],
                "experiment_id": row["experiment_id"],
                "spread_capture": _fmt(row.get("spread_capture"), digits=4),
                "adverse_selection_cost": _fmt(row.get("adverse_selection_cost"), digits=4),
                "inventory_cost": _fmt(row.get("inventory_cost"), digits=4),
                "hedging_cost": _fmt(row.get("hedging_cost"), digits=4),
                "missing_fields": ", ".join(row.get("missing_fields", [])),
            }
        )
    lines.append(
        _format_table(
            rows,
            [
                "strategy",
                "source_file",
                "experiment_id",
                "spread_capture",
                "adverse_selection_cost",
                "inventory_cost",
                "hedging_cost",
                "missing_fields",
            ],
        )
    )
    lines.append("")
    lines.append("## Missing Entries")
    lines.append("")
    lines.append(
        _format_table(report["missing_entries"], ["strategy", "source_file", "missing_fields"])
    )
    lines.append("")
    if report["parse_errors"]:
        lines.append("## Parse Errors")
        lines.append("")
        lines.append(_format_table(report["parse_errors"], ["file", "error"]))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly PnL attribution table.")
    parser.add_argument("--results-dir", default="results", help="Directory for backtest outputs.")
    parser.add_argument("--pattern", default="backtest*.json", help="Glob pattern in results dir.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Optional explicit input JSON files. If set, results-dir/pattern is ignored.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/weekly-pnl-attribution.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/weekly-pnl-attribution.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when no strategy rows are extracted.",
    )
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    if args.inputs:
        input_files = [Path(p).resolve() for p in args.inputs]
    else:
        results_dir = (repo_root / args.results_dir).resolve()
        input_files = _discover_input_files(results_dir, args.pattern)

    if not input_files:
        print("Weekly PnL attribution: no input files found.")
        return 2 if args.strict else 0

    report = _build_report(input_files)

    md_path = (repo_root / args.output_md).resolve()
    json_path = (repo_root / args.output_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if report["summary"]["strategies"] == 0:
        print("Weekly PnL attribution: no strategy rows extracted.")
        return 2 if args.strict else 0

    print(
        "Weekly PnL attribution: "
        f"{report['summary']['strategies']} strategy row(s), "
        f"{report['summary']['missing_entries']} row(s) with missing fields."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
