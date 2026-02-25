"""
Build a compact weekly Markdown summary from research-audit artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def render_weekly_summary(
    iv_report: dict[str, Any],
    model_report: dict[str, Any],
    drift_report: dict[str, Any],
) -> str:
    """Render one-page weekly summary for CI/GitHub dashboard."""
    iv_summary = iv_report.get("summary", {})
    model = drift_report.get("model_zoo", {})
    iv_drift = drift_report.get("iv_surface", {})
    passed = bool(drift_report.get("passed", False))
    violations = drift_report.get("violations", [])
    best_model = model.get("current_best_model") or model_report.get("results", [{}])[0].get(
        "model", ""
    )

    lines = [
        "# Research Audit Weekly Card",
        "",
        f"- Status: `{'PASS' if passed else 'FAIL'}`",
        f"- Snapshot generated: `{drift_report.get('current_generated_at', '')}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| IV no-arbitrage | `{bool(iv_summary.get('no_arbitrage', False))}` |",
        (
            "| Avg short max-jump reduction | "
            f"`{float(iv_summary.get('avg_max_jump_reduction_short', 0.0)):.6f}` |"
        ),
        (
            "| Baseline short max-jump reduction | "
            f"`{float(iv_drift.get('baseline_avg_max_jump_reduction_short', 0.0)):.6f}` |"
        ),
        (
            "| Short max-jump reduction drop | "
            f"`{float(iv_drift.get('avg_max_jump_reduction_drop_pct', 0.0)):.6f}%` |"
        ),
        f"| Best model | `{best_model}` |",
        f"| Best RMSE | `{float(model.get('current_best_rmse', 0.0)):.6f}` |",
        (
            "| Best RMSE increase vs baseline | "
            f"`{float(model.get('best_rmse_increase_pct', 0.0)):.6f}%` |"
        ),
        "",
        "## Violations",
        "",
    ]

    if violations:
        for violation in violations:
            lines.append(f"- {violation}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weekly research-audit summary markdown.")
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
        "--drift-report-json",
        type=str,
        default="artifacts/research-audit-drift-report.json",
        help="Path to drift report JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/research-audit-weekly-summary.md",
        help="Output summary markdown path.",
    )
    args = parser.parse_args()

    markdown = render_weekly_summary(
        iv_report=_load_json(args.iv_report_json),
        model_report=_load_json(args.model_zoo_json),
        drift_report=_load_json(args.drift_report_json),
    )

    directory = os.path.dirname(args.output_md)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as file_obj:
        file_obj.write(markdown)

    print(f"weekly_summary_md={args.output_md}")


if __name__ == "__main__":
    main()
