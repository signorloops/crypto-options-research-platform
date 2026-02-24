"""
Rough-volatility + jumps experiment runner.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Ensure project root is importable when running as a standalone script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.pricing.rough_volatility import RoughVolConfig, RoughVolatilityPricer


def run_mode(mode: str, seed: int) -> dict:
    """Run one pricing experiment mode and return diagnostics."""
    cfg = RoughVolConfig(
        spot=100.0,
        rate=0.0,
        maturity=0.5,
        n_steps=40,
        n_paths=1200,
        hurst=0.1,
        vol_of_vol=1.2,
        initial_variance=0.04,
        correlation=-0.7,
        seed=seed,
        jump_mode=mode,
        jump_intensity=4.0 if mode != "none" else 0.0,
        jump_mean=-0.01,
        jump_std=0.06 if mode != "none" else 0.0,
        variance_jump_scale=0.3,
        jump_excitation=1.6,
        jump_decay=6.0,
    )
    pricer = RoughVolatilityPricer(cfg)
    out = pricer.price_with_confidence_interval(strike=100.0, option_type="call", confidence=0.95)
    return {
        "mode": out["jump_mode"],
        "price": out["price"],
        "ci_low": out["ci_low"],
        "ci_high": out["ci_high"],
        "avg_jump_events_per_path": out["avg_jump_events_per_path"],
        "avg_jump_intensity": out["avg_jump_intensity"],
        "jump_intensity_std": out["jump_intensity_std"],
        "simulation_time_sec": out["simulation_time_sec"],
        "pricing_time_sec": out["pricing_time_sec"],
        "total_time_sec": out["total_time_sec"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rough-volatility jump-mode experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation.")
    args = parser.parse_args()

    rows = [
        run_mode("none", args.seed),
        run_mode("cojump", args.seed),
        run_mode("clustered", args.seed),
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
