"""Reproducible validation for notebooks/01_market_simulation_demo.ipynb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.generators.synthetic import CompleteMarketSimulator
from research.backtest.engine import BacktestEngine, BacktestResult
from strategies.market_making.avellaneda_stoikov import ASConfig, AvellanedaStoikov
from strategies.market_making.naive import NaiveMMConfig, NaiveMarketMaker


def build_notebook_demo_market_data(days: int = 30, seed: int = 42) -> dict[str, Any]:
    """Build a lightweight dataset that covers the notebook's actual demo needs."""
    simulator = CompleteMarketSimulator(seed=seed)
    rng = np.random.default_rng(seed)
    spot = simulator.price_gen.generate(float(days / 365), rng=rng)
    order_books = simulator.ob_sim.generate_time_series(spot, rng=rng)
    option_snapshot = simulator._generate_options_dataset(spot.iloc[[0]].copy())
    return {
        "spot": spot,
        "order_book": simulator._obs_to_df(order_books),
        "options": option_snapshot,
    }


def _result_metrics(result: BacktestResult) -> dict[str, float]:
    return {
        "total_pnl_usd": float(result.total_pnl_usd),
        "sharpe_ratio": float(result.sharpe_ratio),
        "max_drawdown": float(result.max_drawdown),
        "trade_count": int(result.trade_count),
    }


def _build_report(
    *,
    naive_result: BacktestResult,
    as_result: BacktestResult,
    days: int,
    seed: int,
    spot_points: int,
) -> dict[str, Any]:
    return {
        "days": int(days),
        "seed": int(seed),
        "spot_points": int(spot_points),
        "naive": _result_metrics(naive_result),
        "avellaneda_stoikov": _result_metrics(as_result),
    }


def _assert_non_empty_results(report: dict[str, Any]) -> None:
    for strategy_key in ("naive", "avellaneda_stoikov"):
        metrics = report[strategy_key]
        if int(metrics["trade_count"]) <= 0:
            raise ValueError(f"{strategy_key} produced zero trades")


def _plot_results(
    *,
    naive_result: BacktestResult,
    as_result: BacktestResult,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    naive_result.pnl_series.plot(ax=axes[0, 0], label="Naive MM")
    as_result.pnl_series.plot(ax=axes[0, 0], label="A-S Model")
    axes[0, 0].set_title("Cumulative PnL")
    axes[0, 0].set_ylabel("PnL")
    axes[0, 0].legend()

    naive_result.inventory_series.plot(ax=axes[0, 1], label="Naive MM")
    as_result.inventory_series.plot(ax=axes[0, 1], label="A-S Model")
    axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 1].set_title("Inventory Position")
    axes[0, 1].set_ylabel("Position Size")
    axes[0, 1].legend()

    metrics = ["total_pnl_usd", "sharpe_ratio", "max_drawdown", "trade_count"]
    naive_vals = [_result_metrics(naive_result)[metric] for metric in metrics]
    as_vals = [_result_metrics(as_result)[metric] for metric in metrics]
    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 0].bar(x - width / 2, naive_vals, width, label="Naive")
    axes[1, 0].bar(x + width / 2, as_vals, width, label="A-S")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics, rotation=15)
    axes[1, 0].set_title("Strategy Metrics")
    axes[1, 0].legend()

    axes[1, 1].hist(naive_result.pnl_series.diff().dropna(), bins=25, alpha=0.6, label="Naive")
    axes[1, 1].hist(as_result.pnl_series.diff().dropna(), bins=25, alpha=0.6, label="A-S")
    axes[1, 1].set_title("PnL Change Distribution")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_summary(output_path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Notebook 01 Validation Summary",
        "",
        f"- Days: `{report['days']}`",
        f"- Seed: `{report['seed']}`",
        f"- Spot points: `{report['spot_points']}`",
        "",
        "## Naive",
        "",
        f"- Total PnL (USD): `{report['naive']['total_pnl_usd']:.6f}`",
        f"- Sharpe: `{report['naive']['sharpe_ratio']:.6f}`",
        f"- Max drawdown: `{report['naive']['max_drawdown']:.6f}`",
        f"- Trade count: `{report['naive']['trade_count']}`",
        "",
        "## Avellaneda-Stoikov",
        "",
        f"- Total PnL (USD): `{report['avellaneda_stoikov']['total_pnl_usd']:.6f}`",
        f"- Sharpe: `{report['avellaneda_stoikov']['sharpe_ratio']:.6f}`",
        f"- Max drawdown: `{report['avellaneda_stoikov']['max_drawdown']:.6f}`",
        f"- Trade count: `{report['avellaneda_stoikov']['trade_count']}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(output_dir: Path, days: int = 30, seed: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    market_data = build_notebook_demo_market_data(days=days, seed=seed)
    spot = market_data["spot"]

    naive_strategy = NaiveMarketMaker(NaiveMMConfig(spread_bps=20, quote_size=0.5))
    as_strategy = AvellanedaStoikov(ASConfig(gamma=0.1, sigma=0.5, k=1.5, quote_size=0.5))

    naive_result = BacktestEngine(naive_strategy, random_seed=seed).run(spot)
    as_result = BacktestEngine(as_strategy, random_seed=seed).run(spot)

    report = _build_report(
        naive_result=naive_result,
        as_result=as_result,
        days=days,
        seed=seed,
        spot_points=len(spot),
    )
    _assert_non_empty_results(report)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_results(
        naive_result=naive_result,
        as_result=as_result,
        output_path=output_dir / "strategy_comparison.png",
    )
    _write_summary(output_dir / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Notebook 01 market simulation demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/notebooks/01_market_simulation_demo"),
        help="Directory for validation artifacts.",
    )
    parser.add_argument("--days", type=int, default=30, help="Synthetic market horizon in days.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    report = run_validation(output_dir=args.output_dir, days=args.days, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
