"""
Tests for backtest engine.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.types import (
    Fill,
    MarketState,
    OrderBook,
    OrderBookLevel,
    OrderSide,
    Position,
    QuoteAction,
    Trade,
)
from research.backtest.engine import BacktestEngine, RealisticFillSimulator, _calculate_max_drawdown
from research.backtest.engine import FillSimulatorConfig
from strategies.market_making.naive import NaiveMarketMaker


class TestRealisticFillSimulator:
    """Test fill simulation logic."""

    def test_fill_simulator_creation(self):
        """Test fill simulator initialization."""
        sim = RealisticFillSimulator()
        assert sim is not None

    def test_fill_latency_uses_quote_timestamp_not_current_tick(self):
        """Latency gate should compare trade time to quote placement time."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=50.0,
                latency_std_ms=0.0,
                adverse_selection_factor=0.0,
            ),
            rng=np.random.default_rng(123),
        )
        sim._estimate_fill_probability = lambda *args, **kwargs: 1.0  # type: ignore[method-assign]

        t0 = datetime.now(timezone.utc)
        t1 = t0 + timedelta(milliseconds=120)
        order_book = OrderBook(
            timestamp=t1,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=99.95, size=1.0)],
            asks=[OrderBookLevel(price=100.05, size=1.0)],
        )
        market_state = MarketState(
            timestamp=t1,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=[],
        )
        quote = QuoteAction(bid_price=100.0, bid_size=1.0, ask_price=100.2, ask_size=1.0)
        trade = Trade(
            timestamp=t1,
            instrument="SYNTHETIC",
            price=99.99,
            size=0.2,
            side=OrderSide.SELL,
        )

        fill_without_quote_time = sim.simulate_fill(
            quote=quote,
            market_state=market_state,
            next_trades=[trade],
            transaction_cost_bps=0.0,
        )
        fill_with_quote_time = sim.simulate_fill(
            quote=quote,
            market_state=market_state,
            next_trades=[trade],
            quote_timestamp=t0,
            transaction_cost_bps=0.0,
        )

        assert fill_without_quote_time is None
        assert fill_with_quote_time is not None

    def test_fill_simulator_applies_slippage_and_fee_costs(self):
        """Main fill path should accumulate slippage/fee friction."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=0.0,
                # This test quotes the bid exactly at mid with 10 bps fees, so
                # the after-cost edge vs mid is negative; the min-profit gate
                # (BT_MIN_PROFIT_BPS) would rightly reject such a fill. Disable
                # it here to keep testing the fee/slippage cost accumulation.
                min_profit_bps=0.0,
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
        # Transaction fees push the all-in price above the quoted limit...
        assert fill.price > quote.bid_price
        assert sim.transaction_cost_paid > 0
        # ...but a resting buy limit never fills above its own limit price:
        # maker executions are capped at the quote, so slippage against the
        # quote is zero by construction (book walk can only improve it).
        assert sim.slippage_cost == 0.0

    def test_order_book_slippage_uses_maker_side_depth(self):
        """BUY maker fills should not reference ask-side depth."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(base_latency_ms=0.0, latency_std_ms=0.0),
            rng=np.random.default_rng(11),
        )
        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=99.98, size=1.0), OrderBookLevel(price=99.95, size=1.0)],
            asks=[OrderBookLevel(price=120.0, size=1.0), OrderBookLevel(price=121.0, size=1.0)],
        )
        filled_price = sim._apply_order_book_slippage(
            quote_price=100.0,
            trade_size=0.2,
            order_book=order_book,
            side=OrderSide.BUY,
        )
        assert filled_price < 101.0

    def test_maker_buy_never_fills_above_limit_price(self):
        """A resting buy limit fills at its limit price or better, never above."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(base_latency_ms=0.0, latency_std_ms=0.0),
            rng=np.random.default_rng(3),
        )
        now = datetime.now(timezone.utc)
        # Deep, expensive book: walking it would push a taker's VWAP far
        # above the quote. A maker must still fill at <= its limit price.
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=101.0, size=5.0), OrderBookLevel(price=102.0, size=5.0)],
            asks=[OrderBookLevel(price=103.0, size=5.0)],
        )
        filled_price = sim._apply_order_book_slippage(
            quote_price=100.0,
            trade_size=4.0,
            order_book=order_book,
            side=OrderSide.BUY,
        )
        assert filled_price <= 100.0

    def test_maker_sell_never_fills_below_limit_price(self):
        """A resting sell limit fills at its limit price or better, never below."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(base_latency_ms=0.0, latency_std_ms=0.0),
            rng=np.random.default_rng(3),
        )
        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=99.0, size=5.0)],
            asks=[OrderBookLevel(price=99.0, size=5.0), OrderBookLevel(price=98.0, size=5.0)],
        )
        filled_price = sim._apply_order_book_slippage(
            quote_price=100.0,
            trade_size=4.0,
            order_book=order_book,
            side=OrderSide.SELL,
        )
        assert filled_price >= 100.0

    def test_adverse_selection_uses_empirical_trade_size(self):
        """Large-trade classification should use observed flow, not a constant."""

        def _make_sim(recent_sizes, factor):
            # Seed 1: first uniform draw ≈ 0.5118, which lies strictly
            # between `factor` and `2*factor` for factor=0.3, so the
            # large-trade branch (prob 0.6) and the ordinary branch
            # (prob 0.3) return different verdicts.
            sim = RealisticFillSimulator(
                config=FillSimulatorConfig(
                    base_latency_ms=0.0,
                    latency_std_ms=0.0,
                    adverse_selection_factor=factor,
                ),
                rng=np.random.default_rng(1),
            )
            now = datetime.now(timezone.utc)
            order_book = OrderBook(
                timestamp=now,
                instrument="SYNTHETIC",
                bids=[OrderBookLevel(price=99.9, size=1.0)],
                asks=[OrderBookLevel(price=100.1, size=1.0)],
            )
            recent = [
                Trade(
                    timestamp=now,
                    instrument="SYNTHETIC",
                    price=100.0,
                    size=size,
                    side=OrderSide.BUY,
                )
                for size in recent_sizes
            ]
            market_state = MarketState(
                timestamp=now,
                instrument="SYNTHETIC",
                spot_price=100.0,
                order_book=order_book,
                recent_trades=recent,
            )
            return sim, market_state

        trade = Trade(
            timestamp=datetime.now(timezone.utc),
            instrument="SYNTHETIC",
            price=99.95,
            size=0.2,
            side=OrderSide.SELL,
        )

        # Heavy observed flow (mean 5.0): 0.2 is small relative to flow, so
        # the ordinary branch applies (draw 0.51 > 0.3 -> not adverse).
        # Under the old hardcoded 0.1 mean, 0.2 > 3*0.1 would have taken the
        # large-trade branch and reported adverse.
        sim_heavy, state_heavy = _make_sim([5.0, 5.0, 5.0, 5.0], factor=0.3)
        assert sim_heavy._check_adverse_selection(trade, state_heavy) is False

        # Tiny observed flow (mean 0.01): 0.2 is 20x the empirical mean, so
        # the large-trade branch applies (draw 0.51 < 0.6 -> adverse).
        # Under the old hardcoded mean, 0.2 < 3*0.1 would have reported the
        # ordinary branch and not adverse.
        sim_tiny, state_tiny = _make_sim([0.01, 0.01, 0.01, 0.01], factor=0.3)
        assert sim_tiny._check_adverse_selection(trade, state_tiny) is True

    def test_fill_simulator_tracks_adverse_selection_cost(self):
        """Adverse fills should be reflected in simulator cost metrics."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=1.0,
                # With adverse_selection_factor=1.0 every fill is slapped with
                # 10 bps of slippage, pushing the after-cost edge vs mid below
                # the min-profit gate. Disable the gate so this test can keep
                # verifying adverse-selection cost accumulation.
                min_profit_bps=0.0,
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

    def test_fill_simulator_tracks_spread_capture_metrics(self):
        """Filled maker quotes should contribute positive spread-capture totals."""
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=0.0,
            ),
            rng=np.random.default_rng(5),
        )
        sim._estimate_fill_probability = lambda *args, **kwargs: 1.0  # type: ignore[method-assign]

        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=99.95, size=1.0)],
            asks=[OrderBookLevel(price=100.05, size=1.0)],
        )
        market_state = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=[],
        )
        quote = QuoteAction(bid_price=99.95, bid_size=0.5, ask_price=100.15, ask_size=0.5)
        trade = Trade(
            timestamp=now + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.94,
            size=0.5,
            side=OrderSide.SELL,
        )

        fill = sim.simulate_fill(
            quote=quote,
            market_state=market_state,
            next_trades=[trade],
            transaction_cost_bps=0.0,
        )

        assert fill is not None
        assert sim.total_spread_captured > 0
        assert sim.spread_capture_notional == pytest.approx(market_state.spot_price * fill.size)

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

    def test_synthetic_trade_generation_scales_beyond_low_volume_cap(self):
        """High activity events should produce more than 3 synthetic trades occasionally."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy, random_seed=7)
        now = datetime.now(timezone.utc)
        market_state = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=OrderBook(
                timestamp=now,
                instrument="SYNTHETIC",
                bids=[OrderBookLevel(price=99.95, size=1.0)],
                asks=[OrderBookLevel(price=100.05, size=1.0)],
            ),
            recent_trades=[],
        )

        trade_counts = [len(engine._generate_synthetic_trades(market_state, volume=15.0)) for _ in range(200)]
        assert max(trade_counts) > 3

    def test_synthetic_trade_timestamps_span_quote_interval(self):
        """Synthetic trades should be distributed between quote and current event timestamps."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy, random_seed=7)
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(seconds=1)
        market_state = MarketState(
            timestamp=end_ts,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=OrderBook(
                timestamp=end_ts,
                instrument="SYNTHETIC",
                bids=[OrderBookLevel(price=99.95, size=1.0)],
                asks=[OrderBookLevel(price=100.05, size=1.0)],
            ),
            recent_trades=[],
        )

        trades = []
        for _ in range(20):
            trades = engine._generate_synthetic_trades(
                market_state, volume=10.0, start_timestamp=start_ts
            )
            if trades:
                break

        assert trades
        assert all(start_ts <= t.timestamp <= end_ts for t in trades)

    def test_compute_result_preserves_realized_unrealized_breakdown(self):
        """Backtest result should expose realized/unrealized components consistently."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy, fill_simulator=None, initial_crypto_balance=1.0)
        ts = datetime.now(timezone.utc)

        engine.crypto_balance = 1.1
        engine.positions["SYNTHETIC"] = Position("SYNTHETIC", 1.0, 100.0)
        total = engine._calculate_crypto_pnl(90.0)
        engine._pnl_history = [(ts, total)]
        engine._inventory_history = [(ts, 1.0)]
        engine._crypto_balance_history = [(ts, engine.crypto_balance)]

        result = engine._compute_result(current_price=90.0)

        expected_realized = 0.1
        # Unrealized PnL is mark-to-market inventory value in crypto terms:
        # size * (current - entry) / current = 1.0 * (90 - 100) / 90
        expected_unrealized = 1.0 * (90.0 - 100.0) / 90.0
        assert result.realized_pnl == pytest.approx(expected_realized)
        assert result.unrealized_pnl == pytest.approx(expected_unrealized)
        assert result.inventory_pnl == pytest.approx(expected_unrealized)
        assert result.total_pnl_crypto == pytest.approx(expected_realized + expected_unrealized)

    def test_compute_result_exposes_quote_fill_rate_and_spread_capture(self):
        """Computed results should expose quote-derived fill rate and spread capture metrics."""
        strategy = NaiveMarketMaker()
        fill_simulator = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=0.0,
            ),
            rng=np.random.default_rng(9),
        )
        engine = BacktestEngine(strategy, fill_simulator=fill_simulator, initial_crypto_balance=1.0)
        ts = datetime.now(timezone.utc)

        engine.quotes = [
            QuoteAction(bid_price=100.0, bid_size=1.0, ask_price=100.2, ask_size=1.0)
            for _ in range(4)
        ]
        engine.trades = [
            Fill(timestamp=ts, instrument="SYNTHETIC", side=OrderSide.BUY, price=100.0, size=0.2),
            Fill(timestamp=ts, instrument="SYNTHETIC", side=OrderSide.SELL, price=100.2, size=0.2),
        ]
        engine._pnl_history = [(ts, 0.05)]
        engine._inventory_history = [(ts, 0.0)]
        engine._crypto_balance_history = [(ts, engine.crypto_balance)]
        fill_simulator.total_spread_captured = 8.0
        fill_simulator.spread_capture_notional = 2_000.0

        result = engine._compute_result(current_price=100.0)

        assert result.quote_count == 4
        assert result.fill_rate == pytest.approx(0.5)
        assert result.total_spread_captured == pytest.approx(8.0)
        assert result.avg_spread_captured_bps == pytest.approx(40.0)

    def test_quote_uses_current_snapshot_and_fills_against_next_trades(self):
        """Strategy should quote on the current event and fill only on future trades."""

        class SnapshotRecordingStrategy:
            def __init__(self) -> None:
                self.name = "SnapshotRecorder"
                self.observed_spot_prices = []

            def quote(self, state, position):
                self.observed_spot_prices.append(float(state.spot_price))
                mid = state.order_book.mid_price or state.spot_price
                return QuoteAction(mid, 0.0, mid, 0.0, metadata={})

            def on_fill(self, fill, position) -> None:
                return None

            def reset(self) -> None:
                self.observed_spot_prices.clear()

        strategy = SnapshotRecordingStrategy()
        engine = BacktestEngine(strategy, random_seed=42)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "price": [100.0, 110.0, 120.0],
            }
        )

        engine.run(df)

        # Each event quotes on its own (current) snapshot; no extra lag.
        # Look-ahead is prevented on the fill side, which only matches the
        # resting quote against trades in (t_i, t_{i+1}].
        assert strategy.observed_spot_prices == [100.0, 110.0, 120.0]


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

    def test_drawdown_handles_zero_running_peak_without_silent_zero(self):
        """Drawdown should remain negative when equity starts near zero then declines."""
        pnl_series = pd.Series([0.0, -0.10, -0.20, -0.15])
        max_dd = _calculate_max_drawdown(pnl_series)
        assert max_dd == pytest.approx(-0.20)


def test_engine_periods_per_year_infers_frequency_from_datetime_index():
    """Annualization helper should distinguish daily vs hourly frequencies."""
    daily_idx = pd.date_range("2026-01-01", periods=6, freq="1D")
    hourly_idx = pd.date_range("2026-01-01", periods=6, freq="1h")

    daily = BacktestEngine._periods_per_year(daily_idx, len(daily_idx))
    hourly = BacktestEngine._periods_per_year(hourly_idx, len(hourly_idx))

    assert daily == pytest.approx(365.25, rel=0.05)
    assert hourly == pytest.approx(365.25 * 24.0, rel=0.05)
    assert hourly > daily


class TestBacktestEngineInputSanitization:
    """Regression tests for price-column sanitization (NaN guard)."""

    def test_nan_prices_are_forward_filled_with_warning(self, caplog):
        """A NaN price must not poison the book, trades, or PnL series."""
        strategy = NaiveMarketMaker()
        engine = BacktestEngine(strategy, random_seed=1)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "price": [100.0, np.nan, 102.0, np.inf, 104.0],
            }
        )

        with caplog.at_level("WARNING", logger="research.backtest.engine"):
            result = engine.run(df)

        # The two invalid values were patched to the last valid price.
        assert np.isfinite(result.total_pnl_crypto)
        assert np.isfinite(result.sharpe_ratio)
        assert any("non-finite price" in message for message in caplog.messages)

    def test_all_nan_prices_raise_instead_of_silent_poisoning(self):
        """A price column with no finite values must fail loudly."""
        engine = BacktestEngine(NaiveMarketMaker())
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "price": [np.nan, np.nan, np.nan],
            }
        )
        with pytest.raises(ValueError, match="no finite values"):
            engine.run(df)


class TestBacktestEngineOrderBook:
    """Regression tests for synthetic order book maintenance."""

    def test_update_order_book_uses_event_timestamp_and_relative_spread(self):
        """Book timestamp must advance and spread must scale with price."""
        engine = BacktestEngine(NaiveMarketMaker())
        initial = engine._create_dummy_order_book(100.0)
        event_ts = datetime(2024, 1, 1, 12, 0, 0)

        doubled = engine._update_order_book(initial, 200.0, timestamp=event_ts)

        assert doubled.timestamp == event_ts
        # Relative spread (10 bps) is preserved instead of pinning the
        # initial absolute spread as price moves.
        assert doubled.spread == pytest.approx(200.0 * 0.001)
        initial_relative = initial.spread / 100.0
        assert doubled.spread / 200.0 == pytest.approx(initial_relative)


class TestBacktestEngineHistoryTruncation:
    """Regression test for the history-cap truncation warning."""

    def test_history_truncation_emits_warning(self, caplog):
        """Dropping the oldest history points must be surfaced to callers."""
        engine = BacktestEngine(NaiveMarketMaker(), random_seed=3)
        engine._history_sampling_interval = 1  # record on every tick
        engine._max_history_points = 5
        base_ts = datetime(2024, 1, 1)

        with caplog.at_level("WARNING", logger="research.backtest.engine"):
            for i in range(12):
                engine._record_state(base_ts.replace(minute=i), 100.0)

        assert len(engine._pnl_history) <= 5 + 1  # cap plus in-flight sample
        assert any("exceeded" in message and "truncated" in message for message in caplog.messages)


class TestBacktestEngineFlowDrivenTrades:
    """Regression test for order-flow-driven synthetic trade direction."""

    def test_synthetic_trade_direction_follows_order_flow(self):
        """Buy-side flow must bias subsequent synthetic trades toward buys."""
        engine = BacktestEngine(NaiveMarketMaker(), random_seed=5)
        now = datetime.now(timezone.utc)
        order_book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            # Symmetric 1x1-lot book: book imbalance is structurally zero.
            bids=[OrderBookLevel(price=99.95, size=1.0)],
            asks=[OrderBookLevel(price=100.05, size=1.0)],
        )
        buy_flow = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=[
                Trade(timestamp=now, instrument="SYNTHETIC", price=100.0, size=10.0, side=OrderSide.BUY),
                Trade(timestamp=now, instrument="SYNTHETIC", price=100.0, size=10.0, side=OrderSide.BUY),
                Trade(timestamp=now, instrument="SYNTHETIC", price=100.0, size=1.0, side=OrderSide.SELL),
            ],
        )
        sell_flow = MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=order_book,
            recent_trades=[
                Trade(timestamp=now, instrument="SYNTHETIC", price=100.0, size=1.0, side=OrderSide.BUY),
                Trade(timestamp=now, instrument="SYNTHETIC", price=100.0, size=10.0, side=OrderSide.SELL),
                Trade(timestamp=now, instrument="SYNTHETIC", price=100.0, size=10.0, side=OrderSide.SELL),
            ],
        )

        assert engine._current_flow_imbalance(buy_flow) > 0.5
        assert engine._current_flow_imbalance(sell_flow) < -0.5

        def _buy_fraction(state):
            engine._flow_imbalance = 0.0
            counts = {OrderSide.BUY: 0, OrderSide.SELL: 0}
            for _ in range(60):
                for trade in engine._generate_synthetic_trades(state, volume=5.0):
                    counts[trade.side] += 1
            total = counts[OrderSide.BUY] + counts[OrderSide.SELL]
            return counts[OrderSide.BUY] / total if total else 0.5

        buy_frac = _buy_fraction(buy_flow)
        sell_frac = _buy_fraction(sell_flow)
        # Direction must be order-flow-driven, not pure noise around 0.5.
        assert buy_frac > 0.5
        assert sell_frac < 0.5
        assert buy_frac > sell_frac


class TestFillSimulatorMinProfitGate:
    """Regression tests for the min_profit_bps fill gate (BT_MIN_PROFIT_BPS)."""

    @staticmethod
    def _make_sim(min_profit_bps: float) -> RealisticFillSimulator:
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0,
                latency_std_ms=0.0,
                adverse_selection_factor=0.0,
                min_profit_bps=min_profit_bps,
            ),
            rng=np.random.default_rng(1),
        )
        sim._estimate_fill_probability = lambda *args, **kwargs: 1.0  # type: ignore[method-assign]
        return sim

    @staticmethod
    def _market_state():
        now = datetime.now(timezone.utc)
        return MarketState(
            timestamp=now,
            instrument="SYNTHETIC",
            spot_price=100.0,
            order_book=OrderBook(
                timestamp=now,
                instrument="SYNTHETIC",
                bids=[OrderBookLevel(price=99.95, size=1.0)],
                asks=[OrderBookLevel(price=100.05, size=1.0)],
            ),
            recent_trades=[],
        )

    def test_zero_edge_fill_is_rejected_when_gate_enabled(self):
        """A quote resting at mid has no after-cost edge and must not fill."""
        sim = self._make_sim(min_profit_bps=5.0)
        state = self._market_state()
        trade = Trade(
            timestamp=state.timestamp + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.99,
            size=0.5,
            side=OrderSide.SELL,
        )
        quote_at_mid = QuoteAction(bid_price=100.0, bid_size=1.0, ask_price=100.0, ask_size=1.0)

        fill = sim.simulate_fill(
            quote=quote_at_mid, market_state=state, next_trades=[trade], transaction_cost_bps=0.0
        )

        assert fill is None
        assert sim.total_spread_captured == 0.0
        assert sim.spread_capture_notional == 0.0

    def test_positive_edge_fill_passes_gate(self):
        """A quote with after-cost edge above the threshold still fills."""
        sim = self._make_sim(min_profit_bps=5.0)
        state = self._market_state()
        trade = Trade(
            timestamp=state.timestamp + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.85,
            size=0.5,
            side=OrderSide.SELL,
        )
        # Bid at 99.90 (10 bps below mid) with no fees: after-cost edge vs mid
        # clears the 5 bps threshold, and the trade crosses the bid.
        quote_with_edge = QuoteAction(bid_price=99.90, bid_size=1.0, ask_price=100.2, ask_size=1.0)

        fill = sim.simulate_fill(
            quote=quote_with_edge, market_state=state, next_trades=[trade], transaction_cost_bps=0.0
        )

        assert fill is not None
        assert sim.total_spread_captured > 0.0

    def test_disabled_gate_lets_zero_edge_fill_through(self):
        """min_profit_bps <= 0 disables the gate entirely."""
        sim = self._make_sim(min_profit_bps=0.0)
        state = self._market_state()
        trade = Trade(
            timestamp=state.timestamp + timedelta(milliseconds=1),
            instrument="SYNTHETIC",
            price=99.99,
            size=0.5,
            side=OrderSide.SELL,
        )
        quote_at_mid = QuoteAction(bid_price=100.0, bid_size=1.0, ask_price=100.0, ask_size=1.0)

        fill = sim.simulate_fill(
            quote=quote_at_mid, market_state=state, next_trades=[trade], transaction_cost_bps=0.0
        )

        assert fill is not None


class TestFillSimulatorQueuePositionConfig:
    """Regression tests for the queue_position_random config (BT_QUEUE_POSITION_RANDOM)."""

    @staticmethod
    def _book_and_quote():
        now = datetime.now(timezone.utc)
        book = OrderBook(
            timestamp=now,
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=100.0, size=2.0)],
            asks=[OrderBookLevel(price=100.2, size=2.0)],
        )
        quote = QuoteAction(bid_price=100.0, bid_size=0.5, ask_price=100.2, ask_size=0.5)
        return book, quote

    def test_deterministic_queue_uses_full_same_price_depth(self):
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0, latency_std_ms=0.0, queue_position_random=False
            ),
            rng=np.random.default_rng(2),
        )
        book, quote = self._book_and_quote()
        depth = sim._queue_depth_ahead(quote, OrderSide.BUY, book)
        assert depth == pytest.approx(2.0)

    def test_random_queue_uses_random_fraction_of_same_price_depth(self):
        sim = RealisticFillSimulator(
            config=FillSimulatorConfig(
                base_latency_ms=0.0, latency_std_ms=0.0, queue_position_random=True
            ),
            rng=np.random.default_rng(2),
        )
        book, quote = self._book_and_quote()
        depths = [sim._queue_depth_ahead(quote, OrderSide.BUY, book) for _ in range(50)]
        assert all(0.0 <= d <= 2.0 for d in depths)
        # A deterministic fraction (e.g. exactly 1.0 every time) would mean
        # the random placement never took effect.
        assert len(set(np.round(depths, 6))) > 1
