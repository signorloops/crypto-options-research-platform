"""
Tests for market making strategies.
"""
from datetime import datetime, timedelta, timezone

import pytest

from core.types import MarketState, OrderBook, OrderBookLevel, OrderSide, Position, Trade
from strategies.market_making.avellaneda_stoikov import ASConfig, AvellanedaStoikov
from strategies.market_making.naive import NaiveMarketMaker, NaiveMMConfig


class TestNaiveMarketMaker:
    """Test Naive Market Maker strategy."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = NaiveMMConfig(spread_bps=20, quote_size=1.0)
        strategy = NaiveMarketMaker(config)

        assert strategy.name == "NaiveMM"
        assert strategy.config.spread_bps == 20

    def test_basic_quote(self, sample_order_book):
        """Test basic quote generation."""
        strategy = NaiveMarketMaker(NaiveMMConfig(spread_bps=20, quote_size=1.0))

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=50000,
            order_book=sample_order_book,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=0, avg_entry_price=0)

        quote = strategy.quote(state, position)

        # Check spread is approximately 20 bps
        mid = sample_order_book.mid_price
        expected_half_spread = mid * 20 / 10000 / 2

        assert abs(quote.bid_price - (mid - expected_half_spread)) < 1
        assert abs(quote.ask_price - (mid + expected_half_spread)) < 1
        assert quote.bid_size == 1.0
        assert quote.ask_size == 1.0

    def test_inventory_limit_long(self, sample_order_book):
        """Test inventory limit on long position."""
        config = NaiveMMConfig(spread_bps=20, quote_size=1.0, inventory_limit=5.0)
        strategy = NaiveMarketMaker(config)

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=50000,
            order_book=sample_order_book,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=5.0, avg_entry_price=49000)

        quote = strategy.quote(state, position)

        # Should not bid when at limit
        assert quote.bid_size == 0
        assert quote.ask_size == 1.0

    def test_inventory_limit_short(self, sample_order_book):
        """Test inventory limit on short position."""
        config = NaiveMMConfig(spread_bps=20, quote_size=1.0, inventory_limit=5.0)
        strategy = NaiveMarketMaker(config)

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=50000,
            order_book=sample_order_book,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=-5.0, avg_entry_price=51000)

        quote = strategy.quote(state, position)

        # Should not ask when at short limit
        assert quote.bid_size == 1.0
        assert quote.ask_size == 0

    def test_quote_without_orderbook(self):
        """Test error when no valid order book."""
        strategy = NaiveMarketMaker()

        empty_ob = OrderBook(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            bids=[],
            asks=[]
        )

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=0,
            order_book=empty_ob,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=0, avg_entry_price=0)

        with pytest.raises(ValueError, match="Cannot quote without valid order book"):
            strategy.quote(state, position)

    def test_get_internal_state(self):
        """Test getting internal state."""
        strategy = NaiveMarketMaker(NaiveMMConfig(spread_bps=25))
        state = strategy.get_internal_state()

        assert state["spread_bps"] == 25
        assert state["quote_size"] == 1.0


class TestAvellanedaStoikov:
    """Test Avellaneda-Stoikov strategy."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = ASConfig(gamma=0.1, sigma=0.5, k=1.5)
        strategy = AvellanedaStoikov(config)

        assert strategy.name == "AvellanedaStoikov"
        assert strategy.config.gamma == 0.1

    def test_reservation_price_with_inventory(self, sample_order_book):
        """Test reservation price adjusts for inventory."""
        strategy = AvellanedaStoikov(ASConfig(gamma=0.1, sigma=0.5))

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=50000,
            order_book=sample_order_book,
            recent_trades=[]
        )

        # Long position - reservation price should be lower
        long_pos = Position(instrument="BTC-PERPETUAL", size=5.0, avg_entry_price=49000)
        quote_long = strategy.quote(state, long_pos)

        # Short position - reservation price should be higher
        short_pos = Position(instrument="BTC-PERPETUAL", size=-5.0, avg_entry_price=51000)
        quote_short = strategy.quote(state, short_pos)

        # Long position bids lower (more willing to sell)
        assert quote_long.bid_price < quote_short.bid_price
        # Long position asks lower (more aggressive to sell)
        assert quote_long.ask_price < quote_short.ask_price

    def test_spread_widens_with_volatility(self, sample_order_book):
        """Test that spread widens with higher volatility."""
        mid = sample_order_book.mid_price

        # Low volatility config
        low_vol_strategy = AvellanedaStoikov(ASConfig(gamma=0.1, sigma=0.3, k=1.5))
        # High volatility config
        high_vol_strategy = AvellanedaStoikov(ASConfig(gamma=0.1, sigma=0.8, k=1.5))

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=mid,
            order_book=sample_order_book,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=0, avg_entry_price=0)

        low_vol_quote = low_vol_strategy.quote(state, position)
        high_vol_quote = high_vol_strategy.quote(state, position)

        # High vol should have wider spread
        low_vol_spread = low_vol_quote.ask_price - low_vol_quote.bid_price
        high_vol_spread = high_vol_quote.ask_price - high_vol_quote.bid_price

        assert high_vol_spread > low_vol_spread

    def test_metadata_contents(self, sample_order_book):
        """Test that metadata contains expected fields."""
        strategy = AvellanedaStoikov()

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=50000,
            order_book=sample_order_book,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=2.0, avg_entry_price=49000)

        quote = strategy.quote(state, position)

        assert "reservation_price" in quote.metadata
        assert "inventory" in quote.metadata
        assert "spread_component" in quote.metadata
        assert quote.metadata["inventory"] == 2.0

    def test_reset(self):
        """Test strategy reset."""
        from datetime import datetime, timezone
        strategy = AvellanedaStoikov()
        strategy._start_timestamp = datetime.now(timezone.utc)

        strategy.reset()

        assert strategy._start_timestamp is None

    def test_internal_state(self):
        """Test getting internal state."""
        strategy = AvellanedaStoikov(ASConfig(gamma=0.2, sigma=0.6))
        state = strategy.get_internal_state()

        assert state["gamma"] == 0.2
        assert state["sigma"] == 0.6
        assert "time_remaining" in state

    def test_online_calibration_updates_sigma_and_k(self):
        """Online calibration should adapt sigma/k to observed volatility and activity."""
        strategy = AvellanedaStoikov(
            ASConfig(
                gamma=0.1,
                sigma=0.3,
                k=1.0,
                enable_online_calibration=True,
                calibration_window=12,
            )
        )
        initial_sigma = strategy.config.sigma
        initial_k = strategy.config.k
        position = Position(instrument="BTC-PERPETUAL", size=0, avg_entry_price=0)
        base_time = datetime.now(timezone.utc)

        for i in range(20):
            mid = 50000.0 + (220.0 if i % 2 == 0 else -180.0)
            state = MarketState(
                timestamp=base_time + timedelta(seconds=i),
                instrument="BTC-PERPETUAL",
                spot_price=mid,
                order_book=OrderBook(
                    timestamp=base_time + timedelta(seconds=i),
                    instrument="BTC-PERPETUAL",
                    bids=[OrderBookLevel(price=mid - 5.0, size=2.0)],
                    asks=[OrderBookLevel(price=mid + 5.0, size=2.0)],
                ),
                recent_trades=[
                    Trade(
                        timestamp=base_time + timedelta(seconds=i, milliseconds=j),
                        instrument="BTC-PERPETUAL",
                        price=mid + (0.2 * j),
                        size=0.2,
                        side=OrderSide.BUY if j % 2 == 0 else OrderSide.SELL,
                    )
                    for j in range(4)
                ],
            )
            strategy.quote(state, position)

        assert strategy.config.sigma != pytest.approx(initial_sigma)
        assert strategy.config.k != pytest.approx(initial_k)


class TestStrategyComparison:
    """Test strategy comparison framework."""

    def test_strategy_differences(self, sample_order_book):
        """Test that different strategies produce different quotes."""
        strategies = [
            NaiveMarketMaker(NaiveMMConfig(spread_bps=20)),
            AvellanedaStoikov(ASConfig(gamma=0.1, sigma=0.5))
        ]

        state = MarketState(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            spot_price=50000,
            order_book=sample_order_book,
            recent_trades=[]
        )
        position = Position(instrument="BTC-PERPETUAL", size=0, avg_entry_price=0)

        quotes = [s.quote(state, position) for s in strategies]

        # Quotes should be different (at least bid prices)
        assert quotes[0].bid_price != quotes[1].bid_price or quotes[0].ask_price != quotes[1].ask_price
