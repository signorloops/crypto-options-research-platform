#!/usr/bin/env python3
"""Generate a live CEX-vs-DeFi deviation snapshot report."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python scripts/...` execution without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.quote_integration import (
    build_cex_defi_deviation_dataset,
    build_cex_defi_deviation_dataset_live,
)
from execution.research_dashboard import build_cross_market_deviation_report


def _format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = report["summary"]
    sources = report["sources"]
    alerts = report.get("alerts", [])

    lines.append("# Live Deviation Snapshot")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Mode: `{sources.get('mode', 'unknown')}`")
    if sources.get("mode") == "provider":
        lines.append(f"- CEX provider: `{sources.get('cex_provider', '')}`")
        lines.append(f"- Underlying: `{sources.get('underlying', '')}`")
    else:
        lines.append(f"- CEX file: `{sources.get('cex_file', '')}`")
    lines.append(f"- DeFi file: `{sources.get('defi_file', '')}`")
    lines.append(f"- Align tolerance (s): `{sources.get('align_tolerance_seconds', 0)}`")
    lines.append(f"- Threshold (bps): `{summary['threshold_bps']}`")
    lines.append(f"- Rows: `{summary['n_rows']}`")
    lines.append(f"- Alerts: `{summary['n_alerts']}`")
    lines.append(f"- Max abs deviation (bps): `{summary['max_abs_deviation_bps']:.4f}`")
    lines.append(f"- Mean abs deviation (bps): `{summary['mean_abs_deviation_bps']:.4f}`")
    lines.append("")
    lines.append("## Top Alerts")
    lines.append("")
    if not alerts:
        lines.append("_none_")
        return "\n".join(lines)

    header_cols = list(alerts[0].keys())
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
    for row in alerts[:20]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in header_cols) + " |")
    return "\n".join(lines)


async def _build_dataset(
    *,
    cex_file: str,
    cex_provider: str,
    defi_file: str,
    underlying: str,
    align_tolerance_seconds: float,
) -> tuple[dict[str, Any], Any]:
    if cex_file:
        dataset = build_cex_defi_deviation_dataset(
            Path(cex_file),
            Path(defi_file),
            align_tolerance_seconds=align_tolerance_seconds,
        )
        sources = {
            "mode": "file",
            "cex_file": cex_file,
            "defi_file": defi_file,
            "align_tolerance_seconds": float(align_tolerance_seconds),
        }
        return sources, dataset

    dataset = await build_cex_defi_deviation_dataset_live(
        cex_provider,
        Path(defi_file),
        underlying=underlying,
        align_tolerance_seconds=align_tolerance_seconds,
    )
    sources = {
        "mode": "provider",
        "cex_provider": cex_provider,
        "underlying": underlying,
        "defi_file": defi_file,
        "align_tolerance_seconds": float(align_tolerance_seconds),
    }
    return sources, dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate live CEX-vs-DeFi deviation snapshot.")
    parser.add_argument("--cex-file", default="", help="CEX quote file path.")
    parser.add_argument("--cex-provider", default="", help="CEX live provider, e.g. okx.")
    parser.add_argument("--defi-file", default="", help="DeFi quote file path.")
    parser.add_argument("--underlying", default="BTC-USD", help="Underlying for provider mode.")
    parser.add_argument(
        "--align-tolerance-seconds",
        type=float,
        default=60.0,
        help="Max timestamp gap for fallback quote alignment.",
    )
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=300.0,
        help="Deviation alert threshold in bps.",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/live-deviation-snapshot.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/live-deviation-snapshot.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when alert rows exist.",
    )
    args = parser.parse_args()

    cex_file = args.cex_file or os.getenv("CEX_QUOTES_FILE", "")
    cex_provider = args.cex_provider or os.getenv("CEX_QUOTES_PROVIDER", "")
    defi_file = args.defi_file or os.getenv("DEFI_QUOTES_FILE", "")
    if not defi_file:
        print("Live deviation snapshot: missing DeFi source.")
        return 2
    if not cex_file and not cex_provider:
        print("Live deviation snapshot: provide either cex-file or cex-provider.")
        return 2

    try:
        sources, dataset = asyncio.run(
            _build_dataset(
                cex_file=cex_file,
                cex_provider=cex_provider,
                defi_file=defi_file,
                underlying=args.underlying,
                align_tolerance_seconds=float(args.align_tolerance_seconds),
            )
        )
        deviation = build_cross_market_deviation_report(
            dataset, threshold_bps=float(args.threshold_bps)
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Live deviation snapshot: {exc}")
        return 2

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **deviation,
        "sources": {**sources, "rows_aligned": int(len(dataset))},
    }

    md_path = Path(args.output_md).resolve()
    json_path = Path(args.output_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_format_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    n_alerts = int(report["summary"]["n_alerts"])
    if args.strict and n_alerts > 0:
        print(f"Live deviation snapshot: {n_alerts} alert(s) exceed threshold.")
        return 2

    print(f"Live deviation snapshot: rows={report['summary']['n_rows']}, alerts={n_alerts}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
