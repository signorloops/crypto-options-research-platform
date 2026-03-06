"""
Latency benchmark for critical paths.

Measures:
- Greeks calculation
- Risk check (circuit breaker)
- Quote generation
- End-to-end latency

Target: P95 < 100ms
"""

import argparse
import gc
import io
import json
import logging
import sys
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Allow `python scripts/...` execution without package installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.types import Greeks, MarketState, OrderBook, OrderBookLevel, Position
from research.hedging.adaptive_delta import AdaptiveDeltaHedger
from research.pricing.inverse_options import InverseOptionPricer
from research.risk.circuit_breaker import CircuitBreaker, PortfolioState
from research.signals.regime_detector import VolatilityRegimeDetector
from strategies.market_making.integrated_strategy import IntegratedMarketMakingStrategy

DEFAULT_LATENCY_TARGETS_MS = {
    "greeks_calculation": 4.0,
    "circuit_breaker": 2.0,
    "regime_detector": 3.0,
    "adaptive_hedger": 1.0,
    "quote_generation": 10.0,
    "end_to_end": 100.0,
}


@contextmanager
def _suppressed_benchmark_noise(enabled: bool):
    """Silence nested benchmark output without swallowing exceptions."""
    if not enabled:
        yield
        return

    stderr = io.StringIO()
    stdout = io.StringIO()
    logger_names = [
        "research.risk.circuit_breaker",
        "research.signals.regime_detector",
        "hmmlearn",
        "hmmlearn.base",
    ]
    configured = []
    for name in logger_names:
        logger = logging.getLogger(name)
        configured.append((logger, logger.disabled, logger.level, logger.propagate))
        logger.disabled = True
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)

    with (
        warnings.catch_warnings(),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            for logger, disabled, level, propagate in configured:
                logger.disabled = disabled
                logger.setLevel(level)
                logger.propagate = propagate


class LatencyBenchmark:
    """Benchmark latency for critical paths."""

    def __init__(self, iterations: int = 1000):
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.results: Dict[str, List[float]] = {}
        self.last_all_passed: bool = False
        self._quiet: bool = False

    def _log(self, message: str = "") -> None:
        if not self._quiet:
            print(message)

    def _measure(self, func: Callable, *args, **kwargs) -> float:
        """Measure execution time of a function."""
        start = time.perf_counter()
        with _suppressed_benchmark_noise(self._quiet):
            func(*args, **kwargs)
        end = time.perf_counter()
        return (end - start) * 1000  # Convert to milliseconds

    def _warmup_sample_count(self, cap: int) -> int:
        if self.iterations <= 1:
            return 0
        return min(cap, max(1, self.iterations // 10))

    @staticmethod
    def _warmup_regime_detector(detector: VolatilityRegimeDetector, count: int, seed: int) -> None:
        """Warm up regime detector using deterministic synthetic returns."""
        rng = np.random.default_rng(seed)
        for ret in rng.normal(0, 0.001, count):
            detector.update(ret)

    def _benchmark_registry(self) -> Dict[str, Callable[[], Dict]]:
        return {
            "greeks_calculation": self.benchmark_greeks_calculation,
            "circuit_breaker": self.benchmark_circuit_breaker,
            "regime_detector": self.benchmark_regime_detector,
            "adaptive_hedger": self.benchmark_adaptive_hedger,
            "quote_generation": self.benchmark_quote_generation,
            "end_to_end": self.benchmark_end_to_end,
        }

    @staticmethod
    def _run_benchmark_with_gc_isolation(benchmark: Callable[[], Dict]) -> Dict:
        gc.collect()
        return benchmark()

    def benchmark_greeks_calculation(self) -> Dict:
        """Benchmark Greeks calculation."""
        self._log(f"Benchmarking Greeks calculation ({self.iterations} iterations)...")

        latencies = []
        S, K, T, r, sigma = 50000.0, 50000.0, 30 / 365, 0.05, 0.5

        for _ in range(self.iterations):
            latency = self._measure(
                InverseOptionPricer.calculate_price_and_greeks, S, K, T, r, sigma, "call"
            )
            latencies.append(latency)

        self.results["greeks_calculation"] = latencies
        return self._summarize(
            "Greeks Calculation",
            latencies,
            target_ms=DEFAULT_LATENCY_TARGETS_MS["greeks_calculation"],
        )

    def benchmark_circuit_breaker(self) -> Dict:
        """Benchmark circuit breaker risk check."""
        self._log(f"Benchmarking Circuit Breaker ({self.iterations} iterations)...")

        cb = CircuitBreaker()
        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc), positions={}, cash=1000.0, initial_capital=1000.0
        )

        latencies = []
        for _ in range(self.iterations):
            latency = self._measure(cb.check_risk_limits, portfolio)
            latencies.append(latency)

        self.results["circuit_breaker"] = latencies
        return self._summarize(
            "Circuit Breaker",
            latencies,
            target_ms=DEFAULT_LATENCY_TARGETS_MS["circuit_breaker"],
        )

    def benchmark_regime_detector(self) -> Dict:
        """Benchmark regime detector update."""
        self._log(f"Benchmarking Regime Detector ({self.iterations} iterations)...")

        detector = VolatilityRegimeDetector()
        self._warmup_regime_detector(detector, count=50, seed=42)

        rng = np.random.default_rng(43)

        latencies = []
        for _ in range(self.iterations):
            ret = float(rng.normal(0, 0.001))
            latency = self._measure(detector.update, ret)
            latencies.append(latency)

        self.results["regime_detector"] = latencies
        return self._summarize(
            "Regime Detector",
            latencies,
            target_ms=DEFAULT_LATENCY_TARGETS_MS["regime_detector"],
        )

    def benchmark_adaptive_hedger(self) -> Dict:
        """Benchmark adaptive hedger decision."""
        self._log(f"Benchmarking Adaptive Hedger ({self.iterations} iterations)...")

        hedger = AdaptiveDeltaHedger()
        greeks = Greeks(delta=0.5, gamma=0.01, theta=-0.1, vega=0.2)

        # Populate price history
        now = datetime.now(timezone.utc)
        for i in range(20):
            hedger.update_price(now, 50000.0 + i * 10)

        latencies = []
        for i in range(self.iterations):
            current_time = now + timedelta(milliseconds=i)
            latency = self._measure(hedger.should_hedge, current_time, 51000.0, greeks, 1.0)
            latencies.append(latency)

        self.results["adaptive_hedger"] = latencies
        return self._summarize(
            "Adaptive Hedger",
            latencies,
            target_ms=DEFAULT_LATENCY_TARGETS_MS["adaptive_hedger"],
        )

    def benchmark_quote_generation(self) -> Dict:
        """Benchmark full quote generation."""
        self._log(f"Benchmarking Quote Generation ({self.iterations} iterations)...")

        strategy = IntegratedMarketMakingStrategy()
        self._warmup_regime_detector(strategy.regime_detector, count=100, seed=42)

        # Create market state
        timestamp = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=timestamp,
            instrument="BTC-USD",
            bids=[OrderBookLevel(price=49995.0, size=1.0)],
            asks=[OrderBookLevel(price=50005.0, size=1.0)],
        )
        state = MarketState(
            timestamp=timestamp,
            instrument="BTC-USD",
            spot_price=50000.0,
            order_book=order_book,
            recent_trades=[],
        )
        position = Position("BTC-USD", 0, 0)

        latencies = []
        for _ in range(self.iterations):
            latency = self._measure(strategy.quote, state, position)
            latencies.append(latency)

        self.results["quote_generation"] = latencies
        return self._summarize(
            "Quote Generation",
            latencies,
            target_ms=DEFAULT_LATENCY_TARGETS_MS["quote_generation"],
        )

    def benchmark_end_to_end(self) -> Dict:
        """Benchmark end-to-end strategy execution."""
        self._log(f"Benchmarking End-to-End ({self.iterations} iterations)...")

        strategy = IntegratedMarketMakingStrategy()
        self._warmup_regime_detector(strategy.regime_detector, count=100, seed=42)

        timestamp = datetime.now(timezone.utc)

        latencies = []
        for i in range(self.iterations):
            # Vary price slightly to simulate market
            price = 50000.0 + np.sin(i / 100) * 1000

            order_book = OrderBook(
                timestamp=timestamp + timedelta(milliseconds=i),
                instrument="BTC-USD",
                bids=[OrderBookLevel(price=price - 5, size=1.0)],
                asks=[OrderBookLevel(price=price + 5, size=1.0)],
            )
            state = MarketState(
                timestamp=timestamp + timedelta(milliseconds=i),
                instrument="BTC-USD",
                spot_price=price,
                order_book=order_book,
                recent_trades=[],
            )
            position = Position("BTC-USD", np.sin(i / 50) * 5, 50000.0)

            latency = self._measure(strategy.quote, state, position)
            latencies.append(latency)

        self.results["end_to_end"] = latencies
        return self._summarize(
            "End-to-End",
            latencies,
            target_ms=DEFAULT_LATENCY_TARGETS_MS["end_to_end"],
            warmup_samples=self._warmup_sample_count(cap=20),
        )

    def _summarize(
        self,
        name: str,
        latencies: List[float],
        target_ms: float,
        warmup_samples: int = 0,
    ) -> Dict:
        """Summarize benchmark results."""
        latencies_np = np.array(latencies, dtype=float)
        excluded = max(0, min(int(warmup_samples), len(latencies_np) - 1))
        effective = latencies_np[excluded:] if excluded else latencies_np

        summary = {
            "name": name,
            "target_ms": target_ms,
            "mean_ms": np.mean(effective),
            "median_ms": np.median(effective),
            "p50_ms": np.percentile(effective, 50),
            "p95_ms": np.percentile(effective, 95),
            "p99_ms": np.percentile(effective, 99),
            "min_ms": np.min(effective),
            "max_ms": np.max(effective),
            "std_ms": np.std(effective),
            "sample_count": int(len(effective)),
            "samples_excluded": excluded,
            "meets_target": bool(np.percentile(effective, 95) < target_ms),
        }

        return summary

    def run_all_benchmarks(
        self,
        quiet: bool = False,
        selected_benchmarks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run all benchmarks and return results."""
        self._quiet = quiet
        registry = self._benchmark_registry()
        selected = selected_benchmarks or list(registry.keys())

        invalid = [name for name in selected if name not in registry]
        if invalid:
            raise ValueError(f"Unknown benchmarks: {', '.join(invalid)}")

        self._log("=" * 70)
        self._log("Latency Benchmark Suite")
        self._log("=" * 70)
        self._log(f"Iterations per test: {self.iterations}")
        self._log(f"Benchmarks: {', '.join(selected)}")
        self._log()

        results = []
        for name in selected:
            benchmark = registry[name]
            result = self._run_benchmark_with_gc_isolation(benchmark)
            results.append(result)
            self._log()

        df = pd.DataFrame(results)

        # Print summary table
        self._log("=" * 70)
        self._log("Summary")
        self._log("=" * 70)
        self._log(df.to_string(index=False))
        self._log()

        # Overall status
        all_passed = all(r["meets_target"] for r in results)
        self.last_all_passed = all_passed
        self._log("=" * 70)
        if all_passed:
            self._log("✓ ALL BENCHMARKS PASSED")
        else:
            self._log("✗ SOME BENCHMARKS FAILED")
            failed = [r["name"] for r in results if not r["meets_target"]]
            self._log(f"Failed: {', '.join(failed)}")
        self._log("=" * 70)

        return df

    def generate_report(self, save_path: str = None) -> str:
        """Generate detailed benchmark report."""
        lines = []
        lines.append("# Latency Benchmark Report")
        lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"Iterations: {self.iterations}")
        lines.append("\n" + "=" * 70 + "\n")

        for name, latencies in self.results.items():
            lines.append(f"\n## {name.replace('_', ' ').title()}")
            lines.append("-" * 40)

            latencies = np.array(latencies)
            lines.append(f"- Mean: {np.mean(latencies):.3f} ms")
            lines.append(f"- Median: {np.median(latencies):.3f} ms")
            lines.append(f"- P95: {np.percentile(latencies, 95):.3f} ms")
            lines.append(f"- P99: {np.percentile(latencies, 99):.3f} ms")
            lines.append(f"- Min: {np.min(latencies):.3f} ms")
            lines.append(f"- Max: {np.max(latencies):.3f} ms")
            lines.append(f"- Std: {np.std(latencies):.3f} ms")

        report = "\n".join(lines)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)
            self._log(f"\nReport saved to: {save_path}")

        return report


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _build_json_report(results_df: pd.DataFrame) -> dict[str, Any]:
    benchmarks = [
        {key: _json_safe_value(value) for key, value in record.items()}
        for record in results_df.to_dict(orient="records")
    ]
    checks_passed = sum(1 for row in benchmarks if bool(row.get("meets_target")))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "all_passed": bool(all(bool(row.get("meets_target")) for row in benchmarks)),
            "checks_passed": checks_passed,
            "checks_total": len(benchmarks),
        },
        "benchmarks": benchmarks,
    }


def main():
    """Run latency benchmarks."""
    parser = argparse.ArgumentParser(description="Run latency benchmark suite.")
    parser.add_argument("--iterations", type=int, default=1000, help="Iterations per benchmark")
    parser.add_argument(
        "--fail-on-target-miss",
        action="store_true",
        help="Exit with non-zero status when any benchmark misses its P95 target",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="artifacts/performance/latency_benchmark_report.md",
        help="Path to save markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/performance/latency_benchmark_report.json",
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmarks to run, or 'all'",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    benchmark = LatencyBenchmark(iterations=args.iterations)
    selected = None
    if args.benchmarks.strip().lower() != "all":
        selected = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    results_df = benchmark.run_all_benchmarks(quiet=args.quiet, selected_benchmarks=selected)

    # Save report
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.generate_report(save_path=str(report_path))
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_build_json_report(results_df), indent=2), encoding="utf-8")

    if args.fail_on_target_miss and not benchmark.last_all_passed:
        raise SystemExit(1)

    return results_df


if __name__ == "__main__":
    main()
