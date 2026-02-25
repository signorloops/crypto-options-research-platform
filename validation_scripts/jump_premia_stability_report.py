"""
Generate a deterministic jump-premia stability report for research audit.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from research.signals.jump_risk_premia import JumpRiskPremiaEstimator


def build_synthetic_prices(
    seed: int = 42,
    n_points: int = 720,
    start_price: float = 50000.0,
) -> pd.Series:
    """Build deterministic synthetic price path with clustered jump bursts."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.0025, size=n_points)

    # Positive and negative clustered shocks to stress jump-premia estimator.
    rets[120:126] += 0.015
    rets[300:308] -= 0.018
    rets[520:525] += 0.014
    rets[620:627] -= 0.016

    prices = start_price * np.exp(np.cumsum(rets))
    index = pd.date_range(
        start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        periods=n_points,
        freq="min",
    )
    return pd.Series(prices, index=index)


def build_report(
    prices: pd.Series,
    window: int = 96,
    jump_zscore: float = 2.5,
) -> dict[str, Any]:
    """Build jump-premia summary report."""
    estimator = JumpRiskPremiaEstimator(window=window, jump_zscore=jump_zscore)
    frame = estimator.estimate_series_from_prices(prices)
    if frame.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "n_points": int(len(prices)),
                "window": int(window),
                "jump_zscore": float(jump_zscore),
                "latest_net_jump_premium": 0.0,
                "net_jump_premium_std": 0.0,
                "positive_dominance_ratio": 0.0,
                "latest_cluster_imbalance": 0.0,
            },
        }

    net = frame["net_jump_premium"].to_numpy(dtype=float)
    positive_dominance_ratio = float(np.mean(net > 0.0))
    latest = frame.iloc[-1]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "n_points": int(len(prices)),
            "window": int(window),
            "jump_zscore": float(jump_zscore),
            "latest_positive_jump_premium": float(latest["positive_jump_premium"]),
            "latest_negative_jump_premium": float(latest["negative_jump_premium"]),
            "latest_net_jump_premium": float(latest["net_jump_premium"]),
            "net_jump_premium_mean": float(np.mean(net)),
            "net_jump_premium_std": float(np.std(net)),
            "positive_dominance_ratio": positive_dominance_ratio,
            "latest_cluster_imbalance": float(latest["jump_cluster_imbalance"]),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render report markdown."""
    summary = report["summary"]
    lines = [
        "# Jump Premia Stability Report",
        "",
        f"- Generated (UTC): `{report['generated_at']}`",
        f"- Points: `{summary['n_points']}`",
        f"- Estimator window: `{summary['window']}`",
        f"- Jump z-score: `{summary['jump_zscore']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Latest positive jump premium | `{summary.get('latest_positive_jump_premium', 0.0):.6f}` |",
        f"| Latest negative jump premium | `{summary.get('latest_negative_jump_premium', 0.0):.6f}` |",
        f"| Latest net jump premium | `{summary['latest_net_jump_premium']:.6f}` |",
        f"| Net jump premium mean | `{summary.get('net_jump_premium_mean', 0.0):.6f}` |",
        f"| Net jump premium std | `{summary['net_jump_premium_std']:.6f}` |",
        f"| Positive dominance ratio | `{summary['positive_dominance_ratio']:.6f}` |",
        f"| Latest cluster imbalance | `{summary['latest_cluster_imbalance']:.6f}` |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate jump-premia stability report.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic path.")
    parser.add_argument("--n-points", type=int, default=720, help="Number of synthetic points.")
    parser.add_argument("--window", type=int, default=96, help="Estimator rolling window.")
    parser.add_argument("--jump-zscore", type=float, default=2.5, help="Jump detection z-score.")
    parser.add_argument(
        "--min-net-std",
        type=float,
        default=0.0,
        help="Optional minimum required std of net jump premium; fail if violated.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/jump-premia-stability-report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/jump-premia-stability-report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    prices = build_synthetic_prices(seed=args.seed, n_points=args.n_points)
    report = build_report(prices=prices, window=args.window, jump_zscore=args.jump_zscore)
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

    print(f"jump_premia_md={args.output_md}")
    print(f"jump_premia_json={args.output_json}")
    observed_std = float(report["summary"]["net_jump_premium_std"])
    if observed_std < float(args.min_net_std):
        raise SystemExit(
            f"Jump premia stability gate failed: std={observed_std:.6f} < min={float(args.min_net_std):.6f}"
        )


if __name__ == "__main__":
    main()
