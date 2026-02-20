"""
Tests for long-horizon research algorithms.
"""
from __future__ import annotations

import numpy as np

from research.execution.almgren_chriss import AlmgrenChrissConfig, AlmgrenChrissExecutor
from research.hedging.deep_hedging import DeepHedger, DeepHedgingConfig
from research.pricing.rough_volatility import RoughVolConfig, RoughVolatilityPricer
from research.signals.bayesian_changepoint import BOCDConfig, OnlineBayesianChangepointDetector
from research.volatility.implied import VolatilityPoint, VolatilitySurface, black_scholes_price


def _simulate_gbm_paths(
    n_paths: int,
    n_steps: int,
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    z = rng.normal(size=(n_paths, n_steps))
    log_r = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_r, axis=1))
    return paths


class TestAlmgrenChrissExecutor:
    def test_optimal_schedule_conservation_and_monotonicity(self):
        cfg = AlmgrenChrissConfig(total_quantity=100.0, n_steps=16, horizon=1.0)
        model = AlmgrenChrissExecutor(cfg)
        inv = model.optimal_inventory_path()
        sched = model.optimal_trading_schedule()

        assert np.isclose(inv[0], 100.0)
        assert np.isclose(inv[-1], 0.0)
        assert np.all(np.diff(inv) <= 1e-10)
        assert np.isclose(np.sum(sched), 100.0, atol=1e-8)

    def test_higher_risk_aversion_front_loads_execution(self):
        low = AlmgrenChrissExecutor(
            AlmgrenChrissConfig(
                total_quantity=100.0,
                n_steps=20,
                risk_aversion_lambda=1e-8,
                volatility=0.4,
            )
        )
        high = AlmgrenChrissExecutor(
            AlmgrenChrissConfig(
                total_quantity=100.0,
                n_steps=20,
                risk_aversion_lambda=1e-4,
                volatility=0.4,
            )
        )
        s_low = low.optimal_trading_schedule()
        s_high = high.optimal_trading_schedule()
        assert s_high[0] > s_low[0]

    def test_participation_cap_is_enforced(self):
        market_volume = np.full(10, 50.0)
        cfg = AlmgrenChrissConfig(
            total_quantity=100.0,
            n_steps=10,
            max_participation_rate=0.3,
            expected_step_market_volume=market_volume,
        )
        model = AlmgrenChrissExecutor(cfg)
        sched = model.optimal_trading_schedule(enforce_participation=True)
        cap = 0.3 * market_volume
        assert np.all(sched <= cap + 1e-8)
        assert np.isclose(np.sum(sched), 100.0, atol=1e-6)

    def test_cost_decomposition_matches_objective(self):
        cfg = AlmgrenChrissConfig(total_quantity=80.0, n_steps=12, risk_aversion_lambda=1e-5, volatility=0.3)
        model = AlmgrenChrissExecutor(cfg)
        sched = model.optimal_trading_schedule()
        decomp = model.cost_decomposition(sched)
        objective = model.objective_value(sched)
        assert np.isclose(decomp["total_objective"], objective, atol=1e-8)


class TestOnlineBayesianChangepointDetector:
    def test_changepoint_probability_spikes_near_regime_shift(self):
        rng = np.random.default_rng(123)
        series = np.concatenate(
            [
                rng.normal(0.0, 0.2, size=120),
                rng.normal(2.0, 0.2, size=120),
            ]
        )

        detector = OnlineBayesianChangepointDetector(
            BOCDConfig(hazard_lambda=80.0, max_run_length=300, alert_threshold=0.35)
        )
        cp_probs = np.array([detector.update(float(x)) for x in series])

        around_shift = cp_probs[100:150].max()
        early_window = cp_probs[:80].max()
        assert around_shift > early_window
        assert around_shift > 0.1

    def test_batch_update_and_top_changepoints(self):
        rng = np.random.default_rng(9)
        series = np.concatenate([rng.normal(0.0, 0.1, 80), rng.normal(1.5, 0.1, 80)])
        detector = OnlineBayesianChangepointDetector(
            BOCDConfig(hazard_lambda=70.0, max_run_length=200, alert_threshold=0.2)
        )
        probs = detector.update_batch(series)
        assert len(probs) == len(series)
        tops = detector.top_changepoints(top_k=3, min_probability=0.1)
        assert len(tops) <= 3
        if tops:
            assert tops[0][1] >= 0.1


class TestRoughVolatilityPricer:
    def test_simulate_paths_shape_and_positive_spot(self):
        pricer = RoughVolatilityPricer(
            RoughVolConfig(
                spot=100.0,
                n_paths=128,
                n_steps=24,
                maturity=0.5,
                seed=7,
            )
        )
        spots, vars_path = pricer.simulate_paths()
        assert spots.shape == (128, 25)
        assert vars_path.shape == (128, 25)
        assert np.all(spots > 0)
        assert np.all(vars_path > 0)

    def test_option_price_is_bounded(self):
        pricer = RoughVolatilityPricer(
            RoughVolConfig(
                spot=100.0,
                n_paths=256,
                n_steps=32,
                maturity=1.0,
                rate=0.01,
                seed=11,
            )
        )
        call = pricer.price_european_option(strike=100.0, option_type="call")
        assert call > 0
        assert call < 100.0

    def test_price_confidence_interval_contains_price(self):
        pricer = RoughVolatilityPricer(
            RoughVolConfig(
                spot=100.0,
                n_paths=300,
                n_steps=20,
                maturity=0.75,
                rate=0.0,
                seed=99,
                antithetic_sampling=True,
            )
        )
        out = pricer.price_with_confidence_interval(strike=100.0, option_type="call", confidence=0.95)
        assert out["ci_low"] <= out["price"] <= out["ci_high"]
        assert out["std_error"] >= 0.0


class TestDeepHedger:
    def test_deep_hedger_reduces_hedging_error_vs_unhedged(self):
        paths = _simulate_gbm_paths(
            n_paths=160,
            n_steps=20,
            s0=100.0,
            mu=0.0,
            sigma=0.2,
            maturity=1.0,
            seed=5,
        )
        strike = 100.0
        hedger = DeepHedger(
            DeepHedgingConfig(
                hidden_layer_sizes=(16, 16),
                max_iter=120,
                transaction_cost_bps=0.5,
                seed=5,
            )
        )
        hedger.fit(paths, strike=strike, maturity=1.0, rate=0.0, vol_proxy=0.2)
        stats = hedger.evaluate_hedging_error(paths, strike=strike, maturity=1.0, option_type="call")

        payoff = np.maximum(paths[:, -1] - strike, 0.0)
        unhedged_mae = float(np.mean(np.abs(-payoff)))
        assert np.isfinite(stats["mae_error"])
        assert stats["mae_error"] < unhedged_mae

    def test_deep_hedger_exposes_diagnostics(self):
        paths = _simulate_gbm_paths(
            n_paths=80,
            n_steps=12,
            s0=100.0,
            mu=0.0,
            sigma=0.2,
            maturity=1.0,
            seed=12,
        )
        hedger = DeepHedger(DeepHedgingConfig(max_iter=80, seed=12))
        hedger.fit(paths, strike=100.0, maturity=1.0, rate=0.0, vol_proxy=0.2)
        diag = hedger.diagnostics()
        assert diag["is_fitted"] == 1.0
        assert diag["n_features"] == 4.0


class TestVolSurfaceArbitrageReport:
    def test_arbitrage_report_structure(self):
        surface = VolatilitySurface()
        s0 = 100.0
        r = 0.01
        sigma = 0.2
        strikes = [80, 90, 100, 110, 120]
        expiries = [0.25, 0.5, 1.0]

        for t in expiries:
            for k in strikes:
                price = black_scholes_price(s0, k, t, r, sigma, is_call=True)
                iv = sigma
                surface.add_point(
                    VolatilityPoint(
                        strike=float(k),
                        expiry=float(t),
                        volatility=float(iv),
                        underlying_price=s0,
                        is_call=True,
                    )
                )
                # Keep side-effect parity with real market build path.
                _ = price

        report = surface.detect_arbitrage_opportunities()
        assert "has_arbitrage" in report
        assert "n_findings" in report
        assert "findings" in report
        assert "summary" in report
