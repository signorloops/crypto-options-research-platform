"""Unit tests for XGBoost spread target-cost construction."""

import importlib
import sys
import types

import pandas as pd
import pytest


class _DummyXGBRegressor:
    def __init__(self, *args, **kwargs):
        self.feature_importances_ = []

    def fit(self, X, y):
        self.feature_importances_ = [0.0] * X.shape[1]
        return self

    def predict(self, X):
        return [20.0] * len(X)


if "xgboost" not in sys.modules:
    fake_xgboost = types.ModuleType("xgboost")
    fake_xgboost.XGBRegressor = _DummyXGBRegressor
    sys.modules["xgboost"] = fake_xgboost

XGBoostSpreadStrategy = importlib.import_module(
    "strategies.market_making.xgboost_spread"
).XGBoostSpreadStrategy


class TestXGBoostSpreadTargetCost:
    """Validate target-cost units and edge-case behavior."""

    def test_simulate_cost_uses_consistent_bps_units(self):
        strategy = XGBoostSpreadStrategy()
        historical = pd.DataFrame({"price": [100.0, 100.0, 100.0]})
        outcome = pd.DataFrame({"price": [100.0, 100.1]})  # +10 bps move

        cost_at_20bps = strategy._simulate_cost(historical, spread=20.0, outcome_window=outcome)
        cost_at_5bps = strategy._simulate_cost(historical, spread=5.0, outcome_window=outcome)

        assert cost_at_20bps == pytest.approx(0.0)
        assert cost_at_5bps == pytest.approx(7.5)

    def test_simulate_cost_applies_wide_spread_penalty_in_bps_space(self):
        strategy = XGBoostSpreadStrategy()
        historical = pd.DataFrame({"price": [100.0, 100.0, 100.0]})
        outcome = pd.DataFrame({"price": [100.0, 100.0]})

        cost = strategy._simulate_cost(historical, spread=60.0, outcome_window=outcome)
        assert cost == pytest.approx(1.0)

    def test_simulate_cost_handles_empty_outcome_window_without_nan(self):
        strategy = XGBoostSpreadStrategy()
        historical = pd.DataFrame({"price": [100.0, 100.0, 100.0]})
        outcome = pd.DataFrame({"price": [100.0]})

        cost = strategy._simulate_cost(historical, spread=20.0, outcome_window=outcome)
        assert cost == pytest.approx(0.0)
