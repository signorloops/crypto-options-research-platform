"""Tests for Almgren-Chriss execution simulator reproducibility controls."""

from __future__ import annotations

import numpy as np

from research.execution.almgren_chriss import AlmgrenChrissConfig, AlmgrenChrissExecutor


def _config(random_seed: int | None = None) -> AlmgrenChrissConfig:
    return AlmgrenChrissConfig(
        total_quantity=100.0,
        n_steps=12,
        horizon=1.0,
        temporary_impact_eta=0.1,
        permanent_impact_gamma=0.01,
        volatility=0.2,
        risk_aversion_lambda=1e-6,
        initial_price=100.0,
        random_seed=random_seed,
    )


def test_simulate_execution_costs_reproducible_with_config_seed():
    executor_a = AlmgrenChrissExecutor(_config(random_seed=17))
    executor_b = AlmgrenChrissExecutor(_config(random_seed=17))

    costs_a = executor_a.simulate_execution_costs(n_paths=256)
    costs_b = executor_b.simulate_execution_costs(n_paths=256)

    assert np.allclose(costs_a, costs_b)


def test_simulate_execution_costs_reproducible_with_explicit_rng():
    executor = AlmgrenChrissExecutor(_config(random_seed=None))
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    costs_a = executor.simulate_execution_costs(n_paths=256, rng=rng_a)
    costs_b = executor.simulate_execution_costs(n_paths=256, rng=rng_b)

    assert np.allclose(costs_a, costs_b)
