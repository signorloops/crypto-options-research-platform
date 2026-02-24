"""
Compare research-audit snapshots and enforce drift guard thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def compare_snapshots(
    baseline: dict[str, Any],
    current: dict[str, Any],
    max_best_rmse_increase_pct: float = 25.0,
    max_iv_reduction_drop_pct: float = 30.0,
    allow_best_model_change: bool = False,
) -> dict[str, Any]:
    """Return diff summary + quality-gate violations."""
    base_iv = baseline.get("iv_surface", {})
    curr_iv = current.get("iv_surface", {})
    base_model = baseline.get("model_zoo", {})
    curr_model = current.get("model_zoo", {})

    base_best_rmse = float(base_model.get("best_rmse", 0.0))
    curr_best_rmse = float(curr_model.get("best_rmse", 0.0))
    rmse_increase_pct = 0.0
    if base_best_rmse > 0:
        rmse_increase_pct = (curr_best_rmse - base_best_rmse) / base_best_rmse * 100.0

    base_iv_reduction = float(base_iv.get("avg_max_jump_reduction_short", 0.0))
    curr_iv_reduction = float(curr_iv.get("avg_max_jump_reduction_short", 0.0))
    iv_drop_pct = 0.0
    if base_iv_reduction > 0:
        iv_drop_pct = (base_iv_reduction - curr_iv_reduction) / base_iv_reduction * 100.0

    base_best_model = str(base_model.get("best_model", ""))
    curr_best_model = str(curr_model.get("best_model", ""))
    model_changed = base_best_model.lower() != curr_best_model.lower()

    rough_diff: dict[str, Any] = {}
    base_rough = baseline.get("rough_jump", {})
    curr_rough = current.get("rough_jump", {})
    for mode in sorted(set(base_rough.keys()) | set(curr_rough.keys())):
        base_price = float(base_rough.get(mode, {}).get("price", 0.0))
        curr_price = float(curr_rough.get(mode, {}).get("price", 0.0))
        rough_diff[mode] = {
            "baseline_price": base_price,
            "current_price": curr_price,
            "price_delta": curr_price - base_price,
        }

    violations: list[str] = []
    if model_changed and not allow_best_model_change:
        violations.append(
            f"Best model changed: baseline={base_best_model}, current={curr_best_model}"
        )
    if rmse_increase_pct > max_best_rmse_increase_pct:
        violations.append(
            "Best-model RMSE increase too large: "
            f"{rmse_increase_pct:.4f}% > {max_best_rmse_increase_pct:.4f}%"
        )
    if iv_drop_pct > max_iv_reduction_drop_pct:
        violations.append(
            "Short-maturity stabilization dropped too much: "
            f"{iv_drop_pct:.4f}% > {max_iv_reduction_drop_pct:.4f}%"
        )
    if not bool(curr_iv.get("no_arbitrage", False)):
        violations.append("Current IV snapshot no_arbitrage=false")

    return {
        "baseline_generated_at": baseline.get("generated_at", ""),
        "current_generated_at": current.get("generated_at", ""),
        "model_zoo": {
            "baseline_best_model": base_best_model,
            "current_best_model": curr_best_model,
            "model_changed": model_changed,
            "baseline_best_rmse": base_best_rmse,
            "current_best_rmse": curr_best_rmse,
            "best_rmse_increase_pct": rmse_increase_pct,
        },
        "iv_surface": {
            "baseline_avg_max_jump_reduction_short": base_iv_reduction,
            "current_avg_max_jump_reduction_short": curr_iv_reduction,
            "avg_max_jump_reduction_drop_pct": iv_drop_pct,
            "current_no_arbitrage": bool(curr_iv.get("no_arbitrage", False)),
        },
        "rough_jump": rough_diff,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def render_markdown(diff: dict[str, Any]) -> str:
    """Render snapshot comparison in Markdown."""
    model = diff["model_zoo"]
    iv = diff["iv_surface"]
    lines = [
        "# Research Audit Drift Report",
        "",
        f"- Baseline generated: `{diff['baseline_generated_at']}`",
        f"- Current generated: `{diff['current_generated_at']}`",
        f"- Status: `{'PASS' if diff['passed'] else 'FAIL'}`",
        "",
        "## Model Zoo",
        "",
        f"- Baseline best model: `{model['baseline_best_model']}`",
        f"- Current best model: `{model['current_best_model']}`",
        f"- Best RMSE increase: `{model['best_rmse_increase_pct']:.6f}%`",
        "",
        "## IV Surface",
        "",
        f"- Baseline avg short max-jump reduction: `{iv['baseline_avg_max_jump_reduction_short']:.6f}`",
        f"- Current avg short max-jump reduction: `{iv['current_avg_max_jump_reduction_short']:.6f}`",
        f"- Reduction drop: `{iv['avg_max_jump_reduction_drop_pct']:.6f}%`",
        f"- Current no-arbitrage: `{iv['current_no_arbitrage']}`",
        "",
        "## Violations",
        "",
    ]
    if diff["violations"]:
        for item in diff["violations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare research-audit snapshots.")
    parser.add_argument(
        "--baseline-json",
        type=str,
        default="validation_scripts/fixtures/research_audit_snapshot_baseline.json",
        help="Baseline snapshot JSON path.",
    )
    parser.add_argument(
        "--current-json",
        type=str,
        default="artifacts/research-audit-snapshot.json",
        help="Current snapshot JSON path.",
    )
    parser.add_argument(
        "--max-best-rmse-increase-pct",
        type=float,
        default=25.0,
        help="Max allowed best-model RMSE increase percent.",
    )
    parser.add_argument(
        "--max-iv-reduction-drop-pct",
        type=float,
        default=30.0,
        help="Max allowed drop percent in short-maturity max-jump reduction.",
    )
    parser.add_argument(
        "--allow-best-model-change",
        action="store_true",
        help="Do not fail if benchmark top model changes.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/research-audit-drift-report.json",
        help="Output diff JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/research-audit-drift-report.md",
        help="Output diff Markdown path.",
    )
    args = parser.parse_args()

    baseline = _load_json(args.baseline_json)
    current = _load_json(args.current_json)
    diff = compare_snapshots(
        baseline=baseline,
        current=current,
        max_best_rmse_increase_pct=float(args.max_best_rmse_increase_pct),
        max_iv_reduction_drop_pct=float(args.max_iv_reduction_drop_pct),
        allow_best_model_change=bool(args.allow_best_model_change),
    )
    markdown = render_markdown(diff)

    for output_path, payload in (
        (args.output_json, json.dumps(diff, indent=2, ensure_ascii=False) + "\n"),
        (args.output_md, markdown),
    ):
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(payload)

    print(f"drift_json={args.output_json}")
    print(f"drift_md={args.output_md}")
    if not diff["passed"]:
        for item in diff["violations"]:
            print(f"DRIFT GUARD FAILED: {item}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
