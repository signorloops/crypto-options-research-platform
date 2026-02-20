"""
Tests for backtest engine.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.types import (
    MarketState,
    OrderBook,
    OrderBookLevel,
    OrderSide,
    QuoteAction,
    Trade,
)
from research.backtest.engine import BacktestEngine, RealisticFillSimulator
from research.backtest.engine import FillSimulatorConfig
from strategies.market_making.naive import NaiveMarketMaker


class TestRealisticFillSimulator:
    """Test fill simulation logic."""

    def test_fill_simulator_creation(self):
        """Test fill simulator initialization."""
        sim = RealisticFillSimulator()
        assert sim is not None

    def test_fill_simulator_applies_slippage_and_fee_costs(self):
        """Main fill path should accumulate slippage/fee friction."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=0.0,
            ),
            rng=np.random.default_rng(1),
        )

        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=99.95, size=0.5), OrderBookLevel(price=99.9, size=0.5)],
            asks=[OrderBookLevel(price=100.05, size=0.5), OrderBookLevel(price=100.1, size=0.5)],
        )
        market_state = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=[],
        )
        quote = QuoteAction(bid_price=100.0, bid_size=0.5, ask_price=100.1, ask_size=0.5)
        trade = Trade(
            timestamp=now + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.99,
            size=0.5,
            side=OrderSide.SELL,
        )

        fill = sim.simulate_fill(
            quote=quote,
            market_state=market_state,
            next_trades=[trade],
            transaction_cost_bps=10.0,
        )

        assert fill is not None
        assert fill.price > quote.bid_price
        assert sim.transaction_cost_paid > 0
        assert sim.slippage_cost > 0

    def test_fill_simulator_tracks_adverse_selection_cost(self):
        """Adverse fills should be reflected in simulator cost metrics."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=1.0,
            ),
            rng=np.random.default_rng(7),
        )
        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=99.9, size=1.0)],
            asks=[OrderBookLevel(price=100.1, size=1.0)],
        )
        market_state = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=[],
        )
        quote = QuoteAction(bid_price=100.0, bid_size=1.0, ask_price=100.2, ask_size=1.0)
        trade = Trade(
            timestamp=now + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.95,
            size=0.4,
            side=OrderSide.SELL,
        )

        fill = sim.simulate_fill(
            quote=quote,
            market_state=market_state,
            next_trades=[trade],
            transaction_cost_bps=0.0,
        )

        assert fill is not None
        assert sim.adverse_selection_cost > 0

    def test_fill_probability_decreases_with_queue_depth(self):
        """Deeper queue should reduce modeled fill probability."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(base_latency_ms=0.0, latency_std_ms=0.0),
            rng=np.random.default_rng(42),
        )
        now = datetime.now(timezone.utc)
        quote = QuoteAction(bid_price=100.0, bid_size=0.5, ask_price=100.2, ask_size=0.5)
        trade = Trade(
            timestamp=now + timedelta(milliseconds=2),
            instrument="SYNTHETIC",
            price=99.98,
            size=0.4,
            side=OrderSide.SELL,
        )

        thin_ob = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=100.0, size=0.1)],
            asks=[OrderBookLevel(price=100.2, size=0.1)],
        )
        deep_ob = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=100.2, size=5.0), OrderBookLevel(price=100.0, size=5.0)],
            asks=[OrderBookLevel(price=100.4, size=5.0)],
        )

        thin_state = MarketState(
            timestamp=now, instrument="SYNTHETIC", spot_price=100.0, order_book=thin_ob, recent_trades=[]
        )
        deep_state = MarketState(
            timestamp=now, instrument="SYNTHETIC", spot_price=100.0, order_book=deep_ob, recent_trades=[]
        )

        p_thin = sim._estimate_fill_probability(quote, trade, OrderSide.BUY, thin_state, latency_ms=0.0)
        p_deep = sim._estimate_fill_probability(quote, trade, OrderSide.BUY, deep_state, latency_ms=0.0)
        assert p_thin > p_deep

    def test_fill_probability_increases_with_quote_competitiveness(self):
        """More competitive quotes should get higher fill probability."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(base_latency_ms=0.0, latency_std_ms=0.0),
            rng=np.random.default_rng(7),
        )
        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=100.0, size=1.0), OrderBookLevel(price=99.9, size=2.0)],
            asks=[OrderBookLevel(price=100.2, size=1.0)],
        )
        state = MarketState(
            timestamp=now, instrument="SYNTHETIC", spot_price=100.0, order_book=order_book, recent_trades=[]
        )
        trade = Trade(
            timestamp=now + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.95,
            size=0.3,
            side=OrderSide.SELL,
        )

        quote_aggressive = QuoteAction(bid_price=100.0, bid_size=0.5, ask_price=100.2, ask_size=0.5)
        quote_passive = QuoteAction(bid_price=99.8, bid_size=0.5, ask_price=100.2, ask_size=0.5)

        p_aggr = sim._estimate_fill_probability(
            quote_aggressive, trade, OrderSide.BUY, state, latency_ms=0.0
        )
        p_passive = sim._estimate_fill_probability(
            quote_passive, trade, OrderSide.BUY, state, latency_ms=0.0
        )
        assert p_aggr > p_passive

    def test_fill_probability_penalizes_high_short_horizon_volatility(self):
        """High short-horizon volatility should lower fill confidence."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(base_latency_ms=0.0, latency_std_ms=0.0),
            rng=np.random.default_rng(13),
        )
        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=100.0, size=1.0)],
            asks=[OrderBookLevel(price=100.2, size=1.0)],
        )
        quote = QuoteAction(bid_price=100.0, bid_size=0.5, ask_price=100.2, ask_size=0.5)
        trade = Trade(
            timestamp=now + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.98,
            size=0.2,
            side=OrderSide.SELL,
        )

        low_vol_trades = [
            Trade(
                timestamp=now - timedelta(milliseconds=5 - i),
                instrument="SYNTHETIC",
                price=100.0 + 0.001 * i,
                size=0.1,
                side=OrderSide.BUY,
            )
            for i in range(5)
        ]
        high_vol_trades = [
            Trade(
                timestamp=now - timedelta(milliseconds=5 - i),
                instrument="SYNTHETIC",
                price=100.0 + (0.15 if i % 2 == 0 else -0.15),
                size=0.1,
                side=OrderSide.BUY,
            )
            for i in range(5)
        ]

        low_vol_state = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=low_vol_trades,
        )
        high_vol_state = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=high_vol_trades,
        )

        p_low = sim._estimate_fill_probability(quote, trade, OrderSide.BUY, low_vol_state, latency_ms=0.0)
        p_high = sim._estimate_fill_probability(quote, trade, OrderSide.BUY, high_vol_state, latency_ms=0.0)
        assert p_low > p_high


class TestBacktestEngine:
    """Test backtest engine."""

    def test_engine_creation(self):
        """Test engine initialization (coin-margined)."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)

        assert engine.strategy == strategy
        assert engine.initial_crypto_balance == 1.0  # Default crypto balance

    def test_basic_backtest(self, sample_market_data):
        """Test basic backtest run (coin-margined)."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)

        result = engine.run(sample_market_data['spot'])

        assert result.strategy_name == "NaiveMM"
        assert isinstance(result.total_pnl_crypto, float)
        assert isinstance(result.total_pnl_usd, float)
        assert isinstance(result.trade_count, int)
        assert result.trade_count >= 0
        assert isinstance(result.sharpe_ci_95, tuple)
        assert isinstance(result.drawdown_ci_95, tuple)
        assert isinstance(result.deflated_sharpe_ratio, float)

    def test_multiple_runs_reproducibility(self, sample_market_data):
        """Test that same strategy produces same results on same data."""
        strategy1 = NaiveMarketMaker()
        strategy2 = NaiveMarketMaker()

        engine1 = BacktestEngine(strategy1, random_seed=123)
        engine2 = BacktestEngine(strategy2, random_seed=123)

        result1 = engine1.run(sample_market_data['spot'])
        result2 = engine2.run(sample_market_data['spot'])

        # Results should be similar (might have small random differences in fill sim)
        assert result1.trade_count == result2.trade_count
        assert abs(result1.total_pnl_crypto - result2.total_pnl_crypto) < 1e-6

    def test_backtest_result_summary(self, sample_market_data):
        """Test that result summary works."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)

        result = engine.run(sample_market_data['spot'])
        summary = result.summary()

        assert isinstance(summary, str)
        assert "NaiveMM" in summary
        assert "Coin-Margined" in summary
        # Check crypto PnL is shown
        assert f"{result.total_pnl_crypto:.8f}" in summary

    def test_pnl_series(self, sample_market_data):
        """Test that PnL series is recorded."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)

        result = engine.run(sample_market_data['spot'])

        assert len(result.pnl_series) > 0
        assert isinstance(result.pnl_series, pd.Series)

    def test_prepare_event_volumes_defaults_without_column(self):
        """Event volumes should default to 1 when source data has no volume column."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "price": [100.0, 101.0, 102.0],
            }
        )
        volumes = engine._prepare_event_volumes(df)
        np.testing.assert_allclose(volumes, np.ones(3))

    def test_prepare_event_volumes_sanitizes_invalid_values(self):
        """NaN/negative/infinite volume inputs should be sanitized for simulation."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "price": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [2.0, np.nan, -1.0, np.inf, 0.3],
            }
        )
        volumes = engine._prepare_event_volumes(df)
        np.testing.assert_allclose(volumes, np.array([2.0, 1.0, 0.0, 1.0, 0.3]))


class TestBacktestWithDifferentStrategies:
    """Test backtest with multiple strategies."""

    def test_different_strategies_produce_different_results(self):
        """Test that different strategies produce different results."""
        from data.generators.synthetic import CompleteMarketSimulator
        from strategies.market_making.avellaneda_stoikov import AvellanedaStoikov

        # Generate more data for this test to ensure strategies produce trades
        # Use seed 123 which generates trades (seed 42 may not generate any)
        sim = CompleteMarketSimulator(seed=123)
        market_data = sim.generate(hours=24, include_options=False)

        strategies = [
            NaiveMarketMaker(),
            AvellanedaStoikov()
        ]

        results = []
        for strategy in strategies:
            engine = BacktestEngine(strategy)
            result = engine.run(market_data['spot'])
            results.append(result)

        # Results should be different (at least PnL or trade count)
        # Note: If both strategies produce 0 trades, they may have identical results
        # In that case, we at least verify they both ran without errors
        if results[0].trade_count > 0 or results[1].trade_count > 0:
            assert results[0].total_pnl_crypto != results[1].total_pnl_crypto or \
                   results[0].trade_count != results[1].trade_count, \
                   "Strategies with trades should produce different results"


class TestBacktestRiskMetrics:
    """Test risk metrics calculation."""

    def test_sharpe_calculation(self, sample_market_data):
        """Test Sharpe ratio calculation."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)

        result = engine.run(sample_market_data['spot'])

        assert isinstance(result.sharpe_ratio, float)
        # Sharpe should be finite
        assert not np.isinf(result.sharpe_ratio)

    def test_drawdown_calculation(self, sample_market_data):
        """Test drawdown calculation."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy)

        result = engine.run(sample_market_data['spot'])

        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown <= 0  # Drawdown is negative or zero
