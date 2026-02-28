"""
Tests for Hawkes strategy comparison framework.

验证场景生成器、指标收集器和对比框架的正确性。
"""
import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from core.types import MarketState, OrderBook, OrderBookLevel, OrderSide, Position, Trade

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research.backtest.hawkes_comparison import (
    ScenarioGenerator,
    HawkesMetricsCollector,
    ComprehensiveHawkesComparison,
    HawkesScenarioConfig,
    ScenarioType,
    HawkesSpecificMetrics,
)
from data.generators.hawkes import HawkesProcess, HawkesParameters
from strategies.market_making.hawkes_mm import HawkesIntensityMonitor, HawkesMarketMaker, HawkesMMConfig


class TestScenarioGenerator(unittest.TestCase):
    """Test scenario generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ScenarioGenerator(base_price=50000.0, price_volatility=0.02)

    def test_hawkes_scenario_configs(self):
        """Test that predefined scenarios have valid parameters."""
        for name, config in self.generator.HAWKES_SCENARIOS.items():
            # Check stability condition: alpha < beta
            self.assertLess(
                config.alpha, config.beta,
                f"Scenario {name}: alpha ({config.alpha}) must be < beta ({config.beta})"
            )
            # Check positive parameters
            self.assertGreater(config.mu, 0, f"Scenario {name}: mu must be positive")
            self.assertGreater(config.alpha, 0, f"Scenario {name}: alpha must be positive")
            self.assertGreater(config.beta, 0, f"Scenario {name}: beta must be positive")

    def test_generate_hawkes_scenarios(self):
        """Test generation of Hawkes scenarios."""
        # Use shorter time for faster tests
        original_T = self.generator.HAWKES_SCENARIOS['low_clustering'].T
        for config in self.generator.HAWKES_SCENARIOS.values():
            config.T = 3600.0  # 1 hour for tests

        scenarios = self.generator.generate_hawkes_scenarios(seed_offset=0)

        # Restore original values
        for config in self.generator.HAWKES_SCENARIOS.values():
            config.T = original_T

        # Check all scenarios were generated
        self.assertEqual(len(scenarios), len(self.generator.HAWKES_SCENARIOS))

        for name, df in scenarios.items():
            # Check DataFrame structure
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('price', df.columns)
            self.assertIn('volume', df.columns)
            self.assertIn('intensity', df.columns)

            # Check data validity
            self.assertTrue(all(df['price'] > 0), f"{name}: All prices must be positive")
            self.assertTrue(all(df['volume'] > 0), f"{name}: All volumes must be positive")
            self.assertTrue(all(df['intensity'] >= 0), f"{name}: All intensities must be non-negative")

            # Check index is datetime
            self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_clustering_levels(self):
        """Test that scenarios have expected clustering levels."""
        config_low = self.generator.HAWKES_SCENARIOS['low_clustering']
        config_high = self.generator.HAWKES_SCENARIOS['high_clustering']

        self.assertEqual(config_low.clustering_level, 'low')
        self.assertEqual(config_high.clustering_level, 'high')
        self.assertLess(config_low.branching_ratio, config_high.branching_ratio)

    def test_stress_scenarios(self):
        """Test generation of stress test scenarios."""
        # Use shorter time for faster tests
        original_T = self.generator.HAWKES_SCENARIOS['low_clustering'].T
        for config in self.generator.HAWKES_SCENARIOS.values():
            config.T = 3600.0  # 1 hour for tests

        scenarios = self.generator.generate_stress_scenarios()

        # Restore original values
        for config in self.generator.HAWKES_SCENARIOS.values():
            config.T = original_T

        expected_scenarios = ['volume_spike', 'liquidity_drought', 'inventory_stress']
        for name in expected_scenarios:
            self.assertIn(name, scenarios)
            self.assertIsInstance(scenarios[name], pd.DataFrame)
            self.assertGreater(len(scenarios[name]), 0)

    def test_events_to_market_data(self):
        """Test conversion of events to market data."""
        config = HawkesScenarioConfig(
            name="test",
            mu=0.1,
            alpha=0.4,
            beta=0.8,
            T=3600.0  # 1 hour
        )

        # Generate events
        params = HawkesParameters(mu=config.mu, alpha=config.alpha, beta=config.beta)
        process = HawkesProcess(params)
        events = process.simulate(config.T, seed=42)

        # Convert to market data
        df = self.generator._events_to_market_data(events, config)

        self.assertEqual(len(df), len(events))
        self.assertIn('price', df.columns)
        self.assertIn('volume', df.columns)


class TestHawkesMetricsCollector(unittest.TestCase):
    """Test Hawkes-specific metrics collection."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = HawkesMetricsCollector()
        self.base_time = datetime(2024, 1, 1)

    def test_record_intensity(self):
        """Test recording intensity values."""
        timestamps = [self.base_time + timedelta(minutes=i) for i in range(10)]
        intensities = np.random.exponential(0.5, 10)

        for ts, intensity in zip(timestamps, intensities):
            self.collector.record_intensity(ts, intensity)

        self.assertEqual(len(self.collector.intensity_history), 10)

    def test_record_spread(self):
        """Test recording spread values."""
        timestamps = [self.base_time + timedelta(minutes=i) for i in range(10)]
        spreads = np.random.uniform(5, 50, 10)

        for ts, spread in zip(timestamps, spreads):
            self.collector.record_spread(ts, spread)

        self.assertEqual(len(self.collector.spread_history), 10)

    def test_record_parameters(self):
        """Test recording parameter estimates."""
        timestamps = [self.base_time + timedelta(minutes=i*10) for i in range(5)]

        for ts in timestamps:
            self.collector.record_parameters(
                timestamp=ts,
                mu=0.1 + np.random.normal(0, 0.01),
                alpha=0.4 + np.random.normal(0, 0.05),
                beta=0.8 + np.random.normal(0, 0.05)
            )

        self.assertEqual(len(self.collector.parameter_history), 5)

    def test_compute_metrics_empty(self):
        """Test computing metrics with no data."""
        metrics = self.collector.compute_metrics()

        self.assertIsInstance(metrics, HawkesSpecificMetrics)
        self.assertEqual(metrics.avg_intensity, 0.0)
        self.assertEqual(metrics.intensity_spread_correlation, 0.0)
        self.assertEqual(metrics.adverse_selection_accuracy, 0.0)

    def test_compute_intensity_spread_correlation(self):
        """Test intensity-spread correlation calculation."""
        # Generate negatively correlated data (high intensity -> low spread)
        n_points = 100
        timestamps = [self.base_time + timedelta(minutes=i) for i in range(n_points)]

        intensities = np.random.exponential(0.5, n_points)
        spreads = 50 - 30 * (intensities / max(intensities)) + np.random.normal(0, 2, n_points)
        spreads = np.maximum(spreads, 5)  # Ensure positive

        for ts, intensity in zip(timestamps, intensities):
            self.collector.record_intensity(ts, intensity)

        for ts, spread in zip(timestamps, spreads):
            self.collector.record_spread(ts, spread)

        metrics = self.collector.compute_metrics()

        # Correlation should be negative
        self.assertLess(metrics.intensity_spread_correlation, 0)

    def test_adverse_selection_accuracy(self):
        """Test adverse selection accuracy calculation."""
        # Generate some correct and incorrect predictions
        timestamps = [self.base_time + timedelta(minutes=i) for i in range(10)]

        # 8 correct, 2 incorrect
        predictions = [
            (True, True),   # Correct detection
            (False, False), # Correct non-detection
            (True, True),
            (False, False),
            (True, False),  # Incorrect - false positive
            (False, True),  # Incorrect - false negative
            (True, True),
            (False, False),
            (True, True),
            (False, False),
        ]

        for ts, (detected, actual) in zip(timestamps, predictions):
            self.collector.record_adverse_selection_signal(ts, detected, actual)

        metrics = self.collector.compute_metrics()
        self.assertEqual(metrics.adverse_selection_accuracy, 0.8)

    def test_get_intensity_series(self):
        """Test getting intensity as pandas Series."""
        timestamps = [self.base_time + timedelta(minutes=i) for i in range(10)]
        intensities = np.random.exponential(0.5, 10)

        for ts, intensity in zip(timestamps, intensities):
            self.collector.record_intensity(ts, intensity)

        series = self.collector.get_intensity_series()

        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), 10)
        self.assertIsInstance(series.index, pd.DatetimeIndex)


class TestComprehensiveHawkesComparison(unittest.TestCase):
    """Test comprehensive comparison framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparison = ComprehensiveHawkesComparison(
            initial_capital=100000.0,
            transaction_cost_bps=2.0
        )

    def test_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.comparison.initial_capital, 100000.0)
        self.assertEqual(self.comparison.transaction_cost_bps, 2.0)
        self.assertIsInstance(self.comparison.scenario_generator, ScenarioGenerator)

    def test_jump_risk_premia_columns_attached_when_enabled(self):
        """Comparison framework should attach jump risk premia columns in opt-in mode."""
        comp = ComprehensiveHawkesComparison(
            initial_capital=100000.0,
            transaction_cost_bps=2.0,
            enable_jump_risk_premia_signals=True,
            jump_risk_window=20,
            jump_risk_zscore=2.0,
        )

        idx = pd.date_range("2024-01-01", periods=80, freq="min")
        rets = np.random.normal(0.0, 0.002, size=80)
        rets[30] += 0.02
        rets[50] -= 0.018
        prices = 50000 * np.exp(np.cumsum(rets))
        market_data = pd.DataFrame({"price": prices, "volume": np.full(80, 1.0)}, index=idx)

        enriched = comp._attach_jump_risk_premia_signals(market_data)
        self.assertIn("positive_jump_premium", enriched.columns)
        self.assertIn("negative_jump_premium", enriched.columns)
        self.assertIn("net_jump_premium", enriched.columns)
        self.assertIn("jump_cluster_imbalance", enriched.columns)
        self.assertTrue(np.isfinite(enriched["net_jump_premium"].iloc[-1]))

    def test_generate_default_scenarios(self):
        """Test default scenario generation."""
        scenarios = self.comparison._generate_default_scenarios()

        # Should include Hawkes scenarios and stress scenarios
        expected_hawkes = ['low_clustering', 'medium_clustering', 'high_clustering', 'critical']
        expected_stress = ['volume_spike', 'liquidity_drought', 'inventory_stress']

        for name in expected_hawkes + expected_stress:
            self.assertIn(name, scenarios)

    def test_classify_scenario(self):
        """Test scenario classification."""
        self.assertEqual(
            self.comparison._classify_scenario('low_clustering'),
            ScenarioType.SYNTHETIC_HAWKES
        )
        self.assertEqual(
            self.comparison._classify_scenario('medium_clustering'),
            ScenarioType.SYNTHETIC_HAWKES
        )
        self.assertEqual(
            self.comparison._classify_scenario('volume_spike'),
            ScenarioType.STRESS_TEST
        )
        self.assertEqual(
            self.comparison._classify_scenario('2023_2024'),
            ScenarioType.REAL_HISTORICAL
        )

    def test_aggregate_rankings(self):
        """Test ranking aggregation."""
        # Create mock results
        from research.backtest.hawkes_comparison import ComparisonResult
        from research.backtest.arena import StrategyScorecard

        mock_scorecards = {
            'StrategyA': StrategyScorecard(
                strategy_name='StrategyA',
                total_pnl=1000,
                total_return_pct=0.01,
                annualized_return=0.1,
                sharpe_ratio=1.5,
                sortino_ratio=1.8,
                max_drawdown=0.05,
                calmar_ratio=2.0,
                total_trades=100,
                win_rate=0.6,
                avg_trade_pnl=10,
                avg_win=20,
                avg_loss=15,
                profit_factor=1.5,
                spread_capture=500,
                adverse_selection_cost=100,
                inventory_cost=50,
                fill_rate=0.3,
                daily_pnl_std=100,
                worst_day=-50,
                best_day=100,
            ),
            'StrategyB': StrategyScorecard(
                strategy_name='StrategyB',
                total_pnl=800,
                total_return_pct=0.008,
                annualized_return=0.08,
                sharpe_ratio=1.2,
                sortino_ratio=1.5,
                max_drawdown=0.04,
                calmar_ratio=2.0,
                total_trades=120,
                win_rate=0.55,
                avg_trade_pnl=6.67,
                avg_win=18,
                avg_loss=12,
                profit_factor=1.4,
                spread_capture=400,
                adverse_selection_cost=80,
                inventory_cost=40,
                fill_rate=0.35,
                daily_pnl_std=80,
                worst_day=-40,
                best_day=90,
            ),
        }

        self.comparison.results = {
            'scenario1': ComparisonResult(
                scenario_name='scenario1',
                scenario_type=ScenarioType.SYNTHETIC_HAWKES,
                scorecards=mock_scorecards
            )
        }

        rankings = self.comparison._aggregate_rankings('sharpe_ratio')

        self.assertEqual(len(rankings), 2)
        self.assertEqual(rankings[0][0], 'StrategyA')  # Higher Sharpe
        self.assertEqual(rankings[1][0], 'StrategyB')

    def test_get_comparison_dataframe_empty(self):
        """Test getting comparison DataFrame with no results."""
        df = self.comparison.get_comparison_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


class TestHawkesScenarioConfig(unittest.TestCase):
    """Test Hawkes scenario configuration."""

    def test_branching_ratio_calculation(self):
        """Test branching ratio calculation."""
        config = HawkesScenarioConfig(
            name="test",
            mu=0.1,
            alpha=0.4,
            beta=0.8
        )
        self.assertEqual(config.branching_ratio, 0.5)

    def test_clustering_level_classification(self):
        """Test clustering level classification."""
        test_cases = [
            (0.1, 0.8, 'low'),      # br = 0.125
            (0.4, 0.8, 'medium'),   # br = 0.5
            (0.7, 0.8, 'high'),     # br = 0.875
            (0.9, 1.0, 'critical'), # br = 0.9
        ]

        for alpha, beta, expected in test_cases:
            config = HawkesScenarioConfig(
                name="test",
                mu=0.1,
                alpha=alpha,
                beta=beta
            )
            self.assertEqual(config.clustering_level, expected)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full comparison workflow."""

    def test_end_to_end_scenario_generation(self):
        """Test complete scenario generation workflow."""
        generator = ScenarioGenerator()

        # Generate all scenario types
        hawkes = generator.generate_hawkes_scenarios()
        stress = generator.generate_stress_scenarios()

        # Verify all scenarios have valid data
        all_scenarios = {**hawkes, **stress}
        self.assertGreater(len(all_scenarios), 0)

        for name, df in all_scenarios.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertIn('price', df.columns)
            self.assertIn('volume', df.columns)


class TestMarkedHawkesControls(unittest.TestCase):
    """Focused tests for marked Hawkes monitor and control signals."""

    def test_marked_trade_size_affects_buy_sell_asymmetry(self):
        """Larger buy marks should lift buy intensity relative to sell intensity."""
        monitor = HawkesIntensityMonitor(
            params=HawkesParameters(mu=0.12, alpha=0.35, beta=1.0),
            window_size=200,
        )

        t = 0.0
        for _ in range(30):
            t += 0.5
            monitor.add_trade(timestamp=t, direction=1, size=3.0)
            t += 0.5
            monitor.add_trade(timestamp=t, direction=-1, size=0.5)

        buy_int, sell_int = monitor.get_buy_sell_intensity(t + 0.1)
        self.assertGreater(buy_int, sell_int)

    def test_online_mle_update_returns_stable_parameters(self):
        """Online MLE estimation should produce valid stable Hawkes parameters."""
        rng = np.random.default_rng(123)
        monitor = HawkesIntensityMonitor(
            params=HawkesParameters(mu=0.1, alpha=0.25, beta=0.9),
            window_size=400,
        )

        t = 0.0
        for dt, direction, size in zip(
            rng.exponential(scale=0.7, size=220),
            rng.choice([-1, 1], size=220),
            rng.lognormal(mean=0.0, sigma=0.4, size=220),
        ):
            t += float(dt)
            monitor.add_trade(timestamp=t, direction=int(direction), size=float(size))

        est = monitor.estimate_parameters_online(use_mle=True)
        self.assertIsNotNone(est)
        self.assertGreater(est.mu, 0.0)
        self.assertGreater(est.alpha, 0.0)
        self.assertGreater(est.beta, est.alpha)

    def test_quote_metadata_exposes_control_signals(self):
        """Quote metadata should include Hawkes control signals for spread/skew decisions."""
        strategy = HawkesMarketMaker(
            HawkesMMConfig(
                base_spread_bps=20.0,
                quote_size=1.0,
                inventory_limit=10.0,
                hawkes_mu=0.1,
                hawkes_alpha=0.3,
                hawkes_beta=0.9,
            )
        )

        ts = datetime(2024, 1, 1, 0, 0, 0)
        order_book = OrderBook(
            timestamp=ts,
            instrument="BTC-USD",
            bids=[OrderBookLevel(price=49995.0, size=2.0)],
            asks=[OrderBookLevel(price=50005.0, size=2.0)],
        )
        recent_trades = [
            Trade(
                timestamp=ts + timedelta(seconds=i * 0.2),
                instrument="BTC-USD",
                price=50000.0 + i,
                size=1.5 if i % 2 == 0 else 0.7,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            )
            for i in range(10)
        ]
        state = MarketState(
            timestamp=ts + timedelta(seconds=5),
            instrument="BTC-USD",
            spot_price=50000.0,
            order_book=order_book,
            recent_trades=recent_trades,
        )
        position = Position("BTC-USD", 2.0, 50000.0)
        quote = strategy.quote(state, position)

        self.assertIn("control_signals", quote.metadata)
        self.assertIn("intensity", quote.metadata["control_signals"])
        self.assertIn("flow_imbalance", quote.metadata["control_signals"])

    def test_metrics_collector_workflow(self):
        """Test complete metrics collection workflow."""
        collector = HawkesMetricsCollector()
        base_time = datetime(2024, 1, 1)

        # Simulate a trading session
        n_points = 50
        for i in range(n_points):
            ts = base_time + timedelta(minutes=i)

            # Record intensity (decreasing over time)
            intensity = 0.5 * np.exp(-i / 20) + 0.1
            collector.record_intensity(ts, intensity)

            # Record spread (increasing as intensity decreases)
            spread = 20 + 30 * (1 - np.exp(-i / 20)) + np.random.normal(0, 1)
            collector.record_spread(ts, max(5, spread))

            # Record parameters periodically
            if i % 10 == 0:
                collector.record_parameters(
                    ts,
                    mu=0.1 + np.random.normal(0, 0.005),
                    alpha=0.4 + np.random.normal(0, 0.02),
                    beta=0.8 + np.random.normal(0, 0.02)
                )

        # Compute metrics
        metrics = collector.compute_metrics()

        self.assertIsInstance(metrics, HawkesSpecificMetrics)
        self.assertGreater(metrics.avg_intensity, 0)
        self.assertIsNotNone(metrics.parameter_stability)


if __name__ == '__main__':
    unittest.main()
