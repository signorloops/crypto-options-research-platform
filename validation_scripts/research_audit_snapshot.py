"""
Build a compact machine-readable snapshot from research-audit artifacts.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Ensure project root is importable when running as a standalone script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validation_scripts.io_utils import (
    load_json as _load_json,
    write_json as _write_json,
)


def parse_rough_jump_report(path: str) -> dict[str, dict[str, float]]:
    """Parse rough-jump text table into mode-keyed metrics."""
    with open(path, "r", encoding="utf-8") as file_obj:
        raw = file_obj.read().strip()
    if not raw:
        return {}

    frame = pd.read_csv(io.StringIO(raw), sep=r"\s+")
    out: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        mode = str(row["mode"])
        out[mode] = {
            "price": float(row["price"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "avg_jump_events_per_path": float(row["avg_jump_events_per_path"]),
            "total_time_sec": float(row["total_time_sec"]),
        }
    return out


def build_snapshot(
    iv_report: dict[str, Any],
    model_report: dict[str, Any],
    rough_jump_by_mode: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build compact summary for trend tracking."""
    iv_summary = iv_report.get("summary", {})
    model_results = model_report.get("results", [])
    best_row = model_results[0] if model_results else {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "iv_surface": {
            "no_arbitrage": bool(iv_summary.get("no_arbitrage", False)),
            "avg_max_jump_reduction_short": float(
                iv_summary.get("avg_max_jump_reduction_short", 0.0)
            ),
            "avg_mean_jump_reduction_short": float(
                iv_summary.get("avg_mean_jump_reduction_short", 0.0)
            ),
        },
        "model_zoo": {
            "quotes_source": model_report.get("quotes_source", ""),
            "n_quotes": int(model_report.get("n_quotes", 0)),
            "best_model": str(best_row.get("model", "")),
            "best_rmse": float(best_row.get("rmse", 0.0)) if best_row else 0.0,
            "best_mae": float(best_row.get("mae", 0.0)) if best_row else 0.0,
        },
        "rough_jump": rough_jump_by_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build research-audit snapshot JSON.")
    parser.add_argument(
        "--iv-report-json",
        type=str,
        default="artifacts/iv-surface-stability-report.json",
        help="Path to IV stability report JSON.",
    )
    parser.add_argument(
        "--model-zoo-json",
        type=str,
        default="artifacts/pricing-model-zoo-benchmark.json",
        help="Path to model-zoo benchmark JSON.",
    )
    parser.add_argument(
        "--rough-jump-txt",
        type=str,
        default="artifacts/rough-jump-experiment.txt",
        help="Path to rough-jump text report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/research-audit-snapshot.json",
        help="Output snapshot JSON path.",
    )
    args = parser.parse_args()

    iv_report = _load_json(args.iv_report_json)
    model_report = _load_json(args.model_zoo_json)
    rough_jump = parse_rough_jump_report(args.rough_jump_txt)
    snapshot = build_snapshot(
        iv_report=iv_report, model_report=model_report, rough_jump_by_mode=rough_jump
    )

    _write_json(args.output_json, snapshot)

    print(f"snapshot_json={args.output_json}")


if __name__ == "__main__":
    main()
