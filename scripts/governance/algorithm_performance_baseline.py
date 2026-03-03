"""Generate lightweight performance baselines for core algorithmic paths."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow `python scripts/...` execution without package installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generators.synthetic import CompleteMarketSimulator
from research.backtest.engine import BacktestEngine
from research.risk.var import VaRCalculator
from scripts.governance.report_utils import (
    write_json as _write_json_shared,
    write_markdown as _write_markdown_shared,
)
from strategies.market_making.naive import NaiveMarketMaker


def _write_markdown(path: Path, content: str) -> None:
    _write_markdown_shared(path, content)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_json_shared(path, payload)


def _validate_positive_int(value: int) -> bool:
    return isinstance(value, int) and value > 0


def _validate_positive_float(value: float) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0


def _summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=float)
    return {
        "n": int(len(values)),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def _build_var_inputs(seed: int, n_obs: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    positions = pd.DataFrame({"value": [60_000.0, 25_000.0, -15_000.0]}, index=["BTC", "ETH", "SOL"])
    returns = pd.DataFrame(
        {
            "BTC": rng.normal(0.0, 0.02, n_obs),
            "ETH": rng.normal(0.0, 0.025, n_obs),
            "SOL": rng.normal(0.0, 0.035, n_obs),
        }
    )
    return positions, returns


def _benchmark_var_monte_carlo(
    iterations: int,
    n_simulations: int,
    random_seed: int,
) -> dict[str, Any]:
    calc = VaRCalculator(confidence_level=0.95)
    positions, returns = _build_var_inputs(seed=random_seed)
    timings_ms: list[float] = []

    for i in range(iterations):
        start = time.perf_counter()
        calc.monte_carlo_var(
            positions,
            returns,
            n_simulations=n_simulations,
            random_seed=random_seed + i,
        )
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize_ms(timings_ms)


def _benchmark_backtest_engine(
    iterations: int,
    hours: int,
    random_seed: int,
) -> dict[str, Any]:
    simulator = CompleteMarketSimulator(seed=random_seed)
    market_data = simulator.generate(hours=hours, include_options=False)["spot"]
    timings_ms: list[float] = []

    for i in range(iterations):
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy, random_seed=random_seed + i)
        start = time.perf_counter()
        engine.run(market_data)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize_ms(timings_ms)


def _build_report(
    *,
    var_stats: dict[str, Any],
    backtest_stats: dict[str, Any],
    var_threshold_ms: float,
    backtest_threshold_ms: float,
) -> dict[str, Any]:
    checks = {
        "var_monte_carlo_p95_ok": bool(var_stats["p95_ms"] <= var_threshold_ms),
        "backtest_engine_p95_ok": bool(backtest_stats["p95_ms"] <= backtest_threshold_ms),
    }
    all_passed = all(checks.values())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "all_passed": all_passed,
            "checks_passed": int(sum(1 for ok in checks.values() if ok)),
            "checks_total": len(checks),
        },
        "thresholds_ms": {
            "var_monte_carlo_p95": float(var_threshold_ms),
            "backtest_engine_p95": float(backtest_threshold_ms),
        },
        "checks": checks,
        "metrics": {
            "var_monte_carlo": var_stats,
            "backtest_engine": backtest_stats,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    metrics = report["metrics"]
    lines = [
        "# Algorithm Performance Baseline",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Checks passed: `{report['summary']['checks_passed']}/{report['summary']['checks_total']}`",
        f"- All passed: `{report['summary']['all_passed']}`",
        "",
        "## Thresholds (P95 ms)",
        "",
        f"- VaR Monte Carlo: `{report['thresholds_ms']['var_monte_carlo_p95']:.2f}`",
        f"- Backtest Engine: `{report['thresholds_ms']['backtest_engine_p95']:.2f}`",
        "",
        "## Check Results",
        "",
        f"- var_monte_carlo_p95_ok: `{checks['var_monte_carlo_p95_ok']}`",
        f"- backtest_engine_p95_ok: `{checks['backtest_engine_p95_ok']}`",
        "",
        "## Metrics",
        "",
        "| Path | Mean (ms) | P95 (ms) | P99 (ms) |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| VaR Monte Carlo | {metrics['var_monte_carlo']['mean_ms']:.3f} | "
            f"{metrics['var_monte_carlo']['p95_ms']:.3f} | {metrics['var_monte_carlo']['p99_ms']:.3f} |"
        ),
        (
            f"| Backtest Engine | {metrics['backtest_engine']['mean_ms']:.3f} | "
            f"{metrics['backtest_engine']['p95_ms']:.3f} | {metrics['backtest_engine']['p99_ms']:.3f} |"
        ),
        "",
    ]
    return "\n".join(lines)


def _run_suite(args: argparse.Namespace) -> dict[str, Any]:
    var_stats = _benchmark_var_monte_carlo(
        iterations=args.var_iterations,
        n_simulations=args.var_simulations,
        random_seed=args.seed,
    )
    backtest_stats = _benchmark_backtest_engine(
        iterations=args.backtest_iterations,
        hours=args.backtest_hours,
        random_seed=args.seed,
    )
    return _build_report(
        var_stats=var_stats,
        backtest_stats=backtest_stats,
        var_threshold_ms=args.var_p95_threshold_ms,
        backtest_threshold_ms=args.backtest_p95_threshold_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate algorithm performance baseline report.")
    parser.add_argument(
        "--output-md",
        default="artifacts/algorithm-performance-baseline.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/algorithm-performance-baseline.json",
        help="Output json report path.",
    )
    parser.add_argument("--var-iterations", type=int, default=8, help="VaR benchmark iterations.")
    parser.add_argument(
        "--var-simulations",
        type=int,
        default=5000,
        help="Monte Carlo simulations per VaR benchmark iteration.",
    )
    parser.add_argument("--backtest-iterations", type=int, default=6, help="Backtest benchmark iterations.")
    parser.add_argument("--backtest-hours", type=int, default=24, help="Synthetic hours for each backtest run.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic fixtures.")
    parser.add_argument(
        "--var-p95-threshold-ms",
        type=float,
        default=250.0,
        help="P95 latency threshold for VaR Monte Carlo path.",
    )
    parser.add_argument(
        "--backtest-p95-threshold-ms",
        type=float,
        default=1200.0,
        help="P95 latency threshold for backtest engine path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any performance check fails.",
    )
    args = parser.parse_args()

    validations = [
        ("--var-iterations", _validate_positive_int(args.var_iterations)),
        ("--var-simulations", _validate_positive_int(args.var_simulations)),
        (
            "--backtest-iterations",
            _validate_positive_int(args.backtest_iterations),
        ),
        ("--backtest-hours", _validate_positive_int(args.backtest_hours)),
        (
            "--var-p95-threshold-ms",
            _validate_positive_float(args.var_p95_threshold_ms),
        ),
        (
            "--backtest-p95-threshold-ms",
            _validate_positive_float(args.backtest_p95_threshold_ms),
        ),
    ]
    invalid_flags = [flag for flag, ok in validations if not ok]
    if invalid_flags:
        print(
            "Algorithm performance baseline: invalid positive numeric args: "
            f"{', '.join(invalid_flags)}."
        )
        return 2

    report = _run_suite(args)
    markdown = _render_markdown(report)

    output_md = Path(args.output_md).resolve()
    output_json = Path(args.output_json).resolve()
    _write_markdown(output_md, markdown)
    _write_json(output_json, report)

    print(
        "Algorithm performance baseline: "
        f"var_p95={report['metrics']['var_monte_carlo']['p95_ms']:.3f}ms, "
        f"backtest_p95={report['metrics']['backtest_engine']['p95_ms']:.3f}ms, "
        f"all_passed={report['summary']['all_passed']}."
    )

    if args.strict and not report["summary"]["all_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
