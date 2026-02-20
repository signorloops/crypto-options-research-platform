"""
Quick benchmark for long-term algorithm modules.
"""
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.execution.almgren_chriss import AlmgrenChrissConfig, AlmgrenChrissExecutor
from research.hedging.deep_hedging import DeepHedger, DeepHedgingConfig
from research.pricing.rough_volatility import RoughVolConfig, RoughVolatilityPricer
from research.signals.bayesian_changepoint import BOCDConfig, OnlineBayesianChangepointDetector


def benchmark_almgren_chriss() -> dict:
    t0 = time.perf_counter()
    model = AlmgrenChrissExecutor(
        AlmgrenChrissConfig(
            total_quantity=1000.0,
            n_steps=32,
            volatility=0.3,
            risk_aversion_lambda=1e-5,
        )
    )
    report = model.build_report(n_paths=500)
    t1 = time.perf_counter()
    return {
        "latency_ms": (t1 - t0) * 1000.0,
        "expected_cost": report.expected_cost,
        "simulated_cost_std": report.simulated_cost_std,
    }


def benchmark_rough_vol() -> dict:
    t0 = time.perf_counter()
    pricer = RoughVolatilityPricer(
        RoughVolConfig(
            spot=100.0,
            maturity=1.0,
            n_steps=40,
            n_paths=1200,
            seed=7,
            antithetic_sampling=True,
        )
    )
    out = pricer.price_with_confidence_interval(strike=100.0, option_type="call", confidence=0.95)
    t1 = time.perf_counter()
    return {
        "latency_ms": (t1 - t0) * 1000.0,
        "price": out["price"],
        "std_error": out["std_error"],
    }


def benchmark_bocd() -> dict:
    rng = np.random.default_rng(11)
    series = np.concatenate([rng.normal(0.0, 0.15, 500), rng.normal(1.0, 0.15, 500)])
    t0 = time.perf_counter()
    detector = OnlineBayesianChangepointDetector(
        BOCDConfig(hazard_lambda=120.0, max_run_length=400, alert_threshold=0.2)
    )
    probs = detector.update_batch(series)
    tops = detector.top_changepoints(top_k=3, min_probability=0.15)
    t1 = time.perf_counter()
    return {
        "latency_ms": (t1 - t0) * 1000.0,
        "max_cp_prob": float(np.max(probs)),
        "top_changepoints": tops,
    }


def benchmark_deep_hedging() -> dict:
    rng = np.random.default_rng(5)
    n_paths = 200
    n_steps = 20
    maturity = 1.0
    dt = maturity / n_steps
    z = rng.normal(size=(n_paths, n_steps))
    sigma = 0.2
    spots = np.zeros((n_paths, n_steps + 1))
    spots[:, 0] = 100.0
    spots[:, 1:] = 100.0 * np.exp(np.cumsum((-0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z, axis=1))

    hedger = DeepHedger(DeepHedgingConfig(max_iter=120, seed=5))
    t0 = time.perf_counter()
    hedger.fit(spots, strike=100.0, maturity=maturity, rate=0.0, vol_proxy=0.2)
    stats = hedger.evaluate_hedging_error(spots, strike=100.0, maturity=maturity, option_type="call")
    t1 = time.perf_counter()
    return {
        "latency_ms": (t1 - t0) * 1000.0,
        "hedging_mae": stats["mae_error"],
        "hedging_std": stats["std_error"],
    }


def main() -> None:
    results = {
        "almgren_chriss": benchmark_almgren_chriss(),
        "rough_volatility": benchmark_rough_vol(),
        "bocd": benchmark_bocd(),
        "deep_hedging": benchmark_deep_hedging(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

