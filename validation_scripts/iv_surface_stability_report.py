"""
Generate an IV surface stability and no-arbitrage report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

# Ensure project root is importable when running as a standalone script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.volatility.implied import VolatilitySurface, black_scholes_price
from research.volatility.surface_audit import audit_surface_stability


def _build_synthetic_surface(seed: int) -> VolatilitySurface:
    """Build a synthetic IV surface with noisier short maturities."""
    rng = np.random.default_rng(seed)
    spot = 50000.0
    rate = 0.01
    maturities = [3.0 / 365.0, 7.0 / 365.0, 14.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0]
    moneyness_grid = [0.75, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25]

    strikes: List[float] = []
    expiries: List[float] = []
    market_prices: List[float] = []
    is_calls: List[bool] = []

    for maturity in maturities:
        for moneyness in moneyness_grid:
            strike = spot * moneyness
            log_m = np.log(max(moneyness, 1e-12))

            base_level = 0.48 - 0.08 * min(maturity * 12.0, 1.0)
            smile = 0.17 * abs(log_m) + 0.04 * (log_m**2)
            short_noise = 0.045 if maturity <= 14.0 / 365.0 else 0.010
            vol = float(np.clip(base_level + smile + rng.normal(0.0, short_noise), 0.05, 2.0))

            price = black_scholes_price(spot, strike, maturity, rate, vol, is_call=True)
            strikes.append(float(strike))
            expiries.append(float(maturity))
            market_prices.append(float(price))
            is_calls.append(True)

    surface = VolatilitySurface()
    surface.add_from_market_data(
        strikes=strikes,
        expiries=expiries,
        market_prices=market_prices,
        underlying=spot,
        r=rate,
        is_calls=is_calls,
    )
    return surface


def _render_markdown(report: Dict[str, object]) -> str:
    """Render report dictionary as Markdown."""
    summary = report["summary"]
    rows = report["expiries"]

    lines = [
        "# IV Surface Stability Report",
        "",
        f"- Generated (UTC): `{report['generated_at']}`",
        f"- Short-maturity threshold: `{report['short_maturity_threshold_days']:.1f}` days",
        f"- No-arbitrage status: `{'PASS' if summary['no_arbitrage'] else 'FAIL'}`",
        f"- Butterfly violations: `{summary['butterfly_violations']}`",
        f"- Calendar violations: `{summary['calendar_violations']}`",
        "",
        "## Short-Maturity Stabilization Impact",
        "",
        f"- Buckets audited: `{summary['n_expiries']}`",
        f"- Short buckets: `{summary['short_maturity_buckets']}`",
        f"- Avg max-jump reduction (short): `{summary['avg_max_jump_reduction_short']:.6f}`",
        f"- Avg mean-jump reduction (short): `{summary['avg_mean_jump_reduction_short']:.6f}`",
        "",
        "## Per-Expiry Detail",
        "",
        "| Expiry (days) | Short? | Points | Raw max jump | Stabilized max jump | Max jump reduction |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {expiry_days:.1f} | {is_short_maturity} | {n_points} | "
            "{raw_max_adjacent_jump:.6f} | {stabilized_max_adjacent_jump:.6f} | "
            "{max_jump_reduction:.6f} |".format(
                expiry_days=float(row["expiry_days"]),
                is_short_maturity="yes" if bool(row["is_short_maturity"]) else "no",
                n_points=int(row["n_points"]),
                raw_max_adjacent_jump=float(row["raw_max_adjacent_jump"]),
                stabilized_max_adjacent_jump=float(row["stabilized_max_adjacent_jump"]),
                max_jump_reduction=float(row["max_jump_reduction"]),
            )
        )

    return "\n".join(lines) + "\n"


def evaluate_quality_gates(
    report: Dict[str, object],
    fail_on_arbitrage: bool = False,
    min_short_max_jump_reduction: float = 0.0,
) -> List[str]:
    """Return quality-gate violations for CI usage."""
    summary = report.get("summary", {})
    violations: List[str] = []

    if fail_on_arbitrage and not bool(summary.get("no_arbitrage", False)):
        violations.append("No-arbitrage check failed.")

    observed = float(summary.get("avg_max_jump_reduction_short", 0.0))
    if observed + 1e-12 < float(min_short_max_jump_reduction):
        violations.append(
            "Short-maturity stabilization below threshold: "
            f"observed={observed:.6f}, required>={min_short_max_jump_reduction:.6f}"
        )
    return violations


def _write_text(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IV surface stability report.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic quotes.")
    parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/iv-surface-stability-report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/iv-surface-stability-report.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--fail-on-arbitrage",
        action="store_true",
        help="Exit non-zero when no-arbitrage checks fail.",
    )
    parser.add_argument(
        "--min-short-max-jump-reduction",
        type=float,
        default=0.0,
        help="Minimum required avg max-jump reduction on short maturities.",
    )
    args = parser.parse_args()

    surface = _build_synthetic_surface(seed=args.seed)
    report = audit_surface_stability(surface=surface)

    md = _render_markdown(report)
    _write_text(args.output_md, md)

    json_dir = os.path.dirname(args.output_json)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    print(f"Markdown report: {args.output_md}")
    print(f"JSON report: {args.output_json}")
    print(md)

    violations = evaluate_quality_gates(
        report=report,
        fail_on_arbitrage=args.fail_on_arbitrage,
        min_short_max_jump_reduction=args.min_short_max_jump_reduction,
    )
    if violations:
        for violation in violations:
            print(f"QUALITY GATE FAILED: {violation}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
