"""Validate inverse-power Monte Carlo baseline against closed-form inverse pricing."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from research.pricing.inverse_options import InverseOptionPricer
from research.pricing.inverse_power_options import InversePowerOptionPricer


def build_validation_grid() -> list[dict[str, Any]]:
    """Build deterministic validation grid."""
    spots = [40000.0, 50000.0, 60000.0]
    strikes = [45000.0, 50000.0, 55000.0]
    maturities = [7.0 / 365.0, 30.0 / 365.0, 90.0 / 365.0]
    sigmas = [0.4, 0.6]
    rates = [0.0, 0.02]
    option_types = ["call", "put"]

    rows: list[dict[str, Any]] = []
    for s in spots:
        for k in strikes:
            for t in maturities:
                for sigma in sigmas:
                    for r in rates:
                        for option_type in option_types:
                            rows.append(
                                {
                                    "S": s,
                                    "K": k,
                                    "T": t,
                                    "sigma": sigma,
                                    "r": r,
                                    "option_type": option_type,
                                }
                            )
    return rows


def run_validation(
    n_paths: int = 120_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run power=1 consistency checks against closed-form inverse pricer."""
    grid = build_validation_grid()
    rows = []

    for idx, row in enumerate(grid):
        closed_form = InverseOptionPricer.calculate_price(
            S=row["S"],
            K=row["K"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"],
            option_type=row["option_type"],
        )
        mc = InversePowerOptionPricer.calculate_price(
            S=row["S"],
            K=row["K"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"],
            option_type=row["option_type"],
            power=1.0,
            n_paths=n_paths,
            seed=seed + idx,
        )
        abs_error = abs(mc - closed_form)
        rel_error = abs_error / max(abs(closed_form), 1e-10)

        out = dict(row)
        out.update(
            {
                "closed_form": float(closed_form),
                "mc_power_1": float(mc),
                "abs_error": float(abs_error),
                "rel_error": float(rel_error),
            }
        )
        rows.append(out)

    abs_errors = np.array([r["abs_error"] for r in rows], dtype=float)
    rel_errors = np.array([r["rel_error"] for r in rows], dtype=float)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(rows),
        "n_paths": int(n_paths),
        "seed": int(seed),
        "summary": {
            "max_abs_error": float(np.max(abs_errors)),
            "mean_abs_error": float(np.mean(abs_errors)),
            "p95_abs_error": float(np.percentile(abs_errors, 95)),
            "max_rel_error": float(np.max(rel_errors)),
            "mean_rel_error": float(np.mean(rel_errors)),
        },
        "cases": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render markdown report."""
    s = report["summary"]
    lines = [
        "# Inverse-Power Validation Report",
        "",
        f"- Generated (UTC): `{report['generated_at']}`",
        f"- Cases: `{report['n_cases']}`",
        f"- Monte Carlo paths: `{report['n_paths']}`",
        f"- Seed: `{report['seed']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Max abs error | `{s['max_abs_error']:.8f}` |",
        f"| Mean abs error | `{s['mean_abs_error']:.8f}` |",
        f"| P95 abs error | `{s['p95_abs_error']:.8f}` |",
        f"| Max rel error | `{s['max_rel_error']:.8f}` |",
        f"| Mean rel error | `{s['mean_rel_error']:.8f}` |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate inverse-power pricer against closed-form inverse."
    )
    parser.add_argument("--n-paths", type=int, default=120000, help="Monte Carlo paths per case.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=6e-4,
        help="Fail when max absolute pricing error exceeds this threshold.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/inverse-power-validation-report.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/inverse-power-validation-report.json",
        help="Output json path.",
    )
    args = parser.parse_args()

    report = run_validation(n_paths=args.n_paths, seed=args.seed)
    markdown = render_markdown(report)

    for output_path, payload in (
        (args.output_md, markdown),
        (args.output_json, json.dumps(report, indent=2, ensure_ascii=False) + "\n"),
    ):
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(payload)

    print(f"inverse_power_validation_md={args.output_md}")
    print(f"inverse_power_validation_json={args.output_json}")

    observed = float(report["summary"]["max_abs_error"])
    if observed > float(args.max_abs_error):
        raise SystemExit(
            f"Inverse-power validation failed: max_abs_error={observed:.8f} > threshold={float(args.max_abs_error):.8f}"
        )


if __name__ == "__main__":
    main()
