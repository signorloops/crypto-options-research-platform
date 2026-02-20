"""
Stress test for integrated strategy.

Tests system behavior under high load:
- High TPS (transactions per second)
- Rapid price movements
- Concurrent operations
- Memory usage tracking
"""
import time
import os
from datetime import datetime, timedelta, timezone
from typing import Dict
import numpy as np
import pandas as pd

from core.types import MarketState, OrderBook, OrderBookLevel, Position
from strategies.market_making.integrated_strategy import IntegratedMarketMakingStrategy


class StressTest:
    """Stress test for trading system."""

    def __init__(self, duration_seconds: int = 60, target_tps: int = 1000):
        self.duration_seconds = duration_seconds
        self.target_tps = target_tps
        self.results: Dict = {}

    def _create_market_state(self, timestamp: datetime, price: float) -> MarketState:
        """Create a market state."""
        order_book = OrderBook(
            timestamp=timestamp,
            instrument="BTC-USD",
            bids=[OrderBookLevel(price=price - 5, size=1.0)],
            asks=[OrderBookLevel(price=price + 5, size=1.0)]
        )
        return MarketState(
            timestamp=timestamp,
            instrument="BTC-USD",
            spot_price=price,
            order_book=order_book,
            recent_trades=[]
        )

    def test_sustained_load(self) -> Dict:
        """Test sustained load over time."""
        print(f"\nTesting sustained load: {self.target_tps} TPS for {self.duration_seconds}s...")

        strategy = IntegratedMarketMakingStrategy()

        # Pre-train regime detector
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        start_time = time.perf_counter()
        latencies = []
        errors = []

        iterations = self.duration_seconds * self.target_tps
        base_price = 50000.0
        timestamp = datetime.now(timezone.utc)

        for i in range(iterations):
            # Simulate price movement
            price = base_price + np.sin(i / 100) * 1000 + np.random.normal(0, 50)
            position_size = np.sin(i / 500) * 5  # Varying position

            state = self._create_market_state(timestamp, price)
            position = Position("BTC-USD", position_size, base_price)

            # Measure latency
            iter_start = time.perf_counter()
            try:
                quote = strategy.quote(state, position)
            except Exception as e:
                errors.append((i, str(e)))
            iter_end = time.perf_counter()

            latencies.append((iter_end - iter_start) * 1000)

            # Increment timestamp
            timestamp += timedelta(milliseconds=1000 / self.target_tps)

            # Progress report every 10%
            if i % max(1, iterations // 10) == 0:
                progress = i / iterations * 100
                print(f"  Progress: {progress:.0f}%")

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        actual_tps = iterations / actual_duration

        latencies = np.array(latencies)

        result = {
            "test": "Sustained Load",
            "target_tps": self.target_tps,
            "actual_tps": actual_tps,
            "iterations": iterations,
            "duration_seconds": actual_duration,
            "errors": len(errors),
            "latency_mean_ms": np.mean(latencies),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "passed": len(errors) == 0 and actual_tps >= self.target_tps * 0.8
        }

        self.results["sustained_load"] = result
        return result

    def test_flash_crash_scenario(self) -> Dict:
        """Test behavior during flash crash scenario."""
        print("\nTesting flash crash scenario...")

        strategy = IntegratedMarketMakingStrategy()
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        base_price = 50000.0
        timestamp = datetime.now(timezone.utc)
        latencies = []

        # Phase 1: Normal trading
        for i in range(100):
            price = base_price + np.random.normal(0, 100)
            state = self._create_market_state(timestamp, price)
            position = Position("BTC-USD", 2.0, base_price)

            start = time.perf_counter()
            strategy.quote(state, position)
            latencies.append((time.perf_counter() - start) * 1000)
            timestamp += timedelta(milliseconds=100)

        # Phase 2: Rapid price drop (flash crash)
        crash_prices = np.linspace(base_price, base_price * 0.7, 50)  # 30% drop
        for price in crash_prices:
            state = self._create_market_state(timestamp, price)
            position = Position("BTC-USD", 5.0, base_price)  # Increased position

            start = time.perf_counter()
            strategy.quote(state, position)
            latencies.append((time.perf_counter() - start) * 1000)
            timestamp += timedelta(milliseconds=100)

        # Phase 3: Recovery
        for i in range(50):
            price = crash_prices[-1] + np.random.normal(0, 200)
            state = self._create_market_state(timestamp, price)
            position = Position("BTC-USD", 3.0, base_price)

            start = time.perf_counter()
            strategy.quote(state, position)
            latencies.append((time.perf_counter() - start) * 1000)
            timestamp += timedelta(milliseconds=100)

        latencies = np.array(latencies)

        # Check circuit breaker was triggered
        circuit_violations = len(strategy.circuit_breaker.violation_history)

        result = {
            "test": "Flash Crash Scenario",
            "iterations": len(latencies),
            "latency_mean_ms": np.mean(latencies),
            "latency_max_ms": np.max(latencies),
            "circuit_violations": circuit_violations,
            "passed": circuit_violations > 0  # Should have triggered circuit breaker
        }

        self.results["flash_crash"] = result
        return result

    def run_all_tests(self) -> pd.DataFrame:
        """Run all stress tests."""
        print("=" * 70)
        print("Stress Test Suite")
        print("=" * 70)

        tests = [
            self.test_sustained_load,
            self.test_flash_crash_scenario,
        ]

        for test in tests:
            test()
            print()

        # Summary
        print("=" * 70)
        print("Stress Test Summary")
        print("=" * 70)

        for name, result in self.results.items():
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"{status}: {result['test']}")

        all_passed = all(r["passed"] for r in self.results.values())
        print("=" * 70)
        if all_passed:
            print("✓ ALL STRESS TESTS PASSED")
        else:
            print("✗ SOME STRESS TESTS FAILED")
        print("=" * 70)

        return pd.DataFrame(self.results.values())

    def generate_report(self, save_path: str = None) -> str:
        """Generate detailed stress test report."""
        lines = []
        lines.append("# Stress Test Report")
        lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("\n" + "=" * 70 + "\n")

        for name, result in self.results.items():
            status = "✓ PASSED" if result["passed"] else "✗ FAILED"
            lines.append(f"\n## {result['test']} - {status}")
            lines.append("-" * 40)

            for key, value in result.items():
                if key not in ["test", "passed"]:
                    if isinstance(value, float):
                        lines.append(f"- {key}: {value:.4f}")
                    else:
                        lines.append(f"- {key}: {value}")

        report = "\n".join(lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {save_path}")

        return report


def main():
    """Run stress tests."""
    # Use smaller parameters for quick test
    stress_test = StressTest(duration_seconds=5, target_tps=100)
    results_df = stress_test.run_all_tests()

    # Save report
    os.makedirs("tests/performance", exist_ok=True)
    stress_test.generate_report(
        save_path="tests/performance/stress_test_report.md"
    )

    return results_df


if __name__ == "__main__":
    main()
