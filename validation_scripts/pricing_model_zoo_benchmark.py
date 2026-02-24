"""
Quick benchmark for crypto option pricing model zoo.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is importable when running as a standalone script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.pricing.model_zoo import CryptoOptionModelZoo, OptionQuote


def _build_synthetic_quotes(
    spot: float,
    rate: float,
    sigma: float,
    seed: int,
    n_per_bucket: int,
) -> list[OptionQuote]:
    rng = np.random.default_rng(seed)
    zoo = CryptoOptionModelZoo()
    maturities = [7.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0, 120.0 / 365.0]
    strikes = [0.8 * spot, 0.9 * spot, 1.0 * spot, 1.1 * spot, 1.2 * spot]

    quotes: list[OptionQuote] = []
    for t in maturities:
        for k in strikes:
            for _ in range(max(n_per_bucket, 1)):
                # Use Bates approximation as "market" generator to include both SV and jumps.
                base = zoo.price_option(
                    model="bates",
                    spot=spot,
                    strike=k,
                    maturity=t,
                    rate=rate,
                    sigma=sigma,
                    is_call=True,
                    model_params={
                        "kappa": 1.8,
                        "theta": 0.32,
                        "v0": 0.45,
                        "rho": -0.55,
                        "jump_intensity": 1.2,
                        "jump_mean": -0.06,
                        "jump_std": 0.28,
                    },
                )
                noisy = max(0.0, base * (1.0 + rng.normal(0.0, 0.01)))
                quotes.append(
                    OptionQuote(
                        spot=spot,
                        strike=float(k),
                        maturity=float(t),
                        rate=rate,
                        market_price=float(noisy),
                        is_call=True,
                    )
                )
    return quotes


def run_benchmark(seed: int = 42, n_per_bucket: int = 1) -> pd.DataFrame:
    """Run model zoo benchmark on synthetic option quotes."""
    spot = 50000.0
    rate = 0.02
    sigma = 0.60

    quotes = _build_synthetic_quotes(
        spot=spot,
        rate=rate,
        sigma=sigma,
        seed=seed,
        n_per_bucket=n_per_bucket,
    )
    zoo = CryptoOptionModelZoo()
    table = zoo.benchmark(
        quotes=quotes,
        sigma=sigma,
        model_params_by_model={
            "merton_jump_diffusion": {
                "jump_intensity": 1.0,
                "jump_mean": -0.05,
                "jump_std": 0.25,
            },
            "kou_jump": {
                "jump_intensity": 1.0,
                "p_up": 0.35,
                "eta1": 12.0,
                "eta2": 8.0,
            },
            "heston": {
                "kappa": 1.8,
                "theta": 0.32,
                "v0": 0.45,
                "rho": -0.55,
            },
            "bates": {
                "kappa": 1.8,
                "theta": 0.32,
                "v0": 0.45,
                "rho": -0.55,
                "jump_intensity": 1.2,
                "jump_mean": -0.06,
                "jump_std": 0.28,
            },
        },
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crypto option pricing model-zoo benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data.")
    parser.add_argument(
        "--n-per-bucket",
        type=int,
        default=1,
        help="Number of quotes per strike/maturity bucket.",
    )
    args = parser.parse_args()

    result = run_benchmark(seed=args.seed, n_per_bucket=args.n_per_bucket)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
