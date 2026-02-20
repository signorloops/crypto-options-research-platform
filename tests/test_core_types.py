"""
Tests for core types module.
"""
from datetime import datetime

import pytest

from core.types import (
    Fill,
    Greeks,
    OptionType,
    OrderSide,
    Position,
    Tick,
    Trade,
)


class TestTick:
    """Test Tick dataclass."""

    def test_tick_creation(self):
        """Test basic tick creation."""
        tick = Tick(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            instrument="BTC-PERPETUAL",
            bid=50000,
            ask=50010,
            bid_size=1.5,
            ask_size=1.0
        )

        assert tick.mid == 50005
        assert tick.spread == 10
        assert tick.spread_bps == 10 / 50005 * 10000

    def test_tick_zero_mid(self):
        """Test tick with zero mid price."""
        tick = Tick(
            timestamp=datetime(2024, 1, 1),
            instrument="TEST",
            bid=0,
            ask=0,
            bid_size=1.0,
            ask_size=1.0
        )

        assert tick.mid == 0
        assert tick.spread == 0
        assert tick.spread_bps == 0


class TestOrderBook:
    """Test OrderBook dataclass."""

    def test_order_book_basic(self, sample_order_book):
        """Test basic order book properties."""
        ob = sample_order_book

        assert ob.best_bid == 50000
        assert ob.best_ask == 50010
        assert ob.mid_price == 50005
        assert ob.spread == 10

    def test_order_book_imbalance(self, sample_order_book):
        """Test order book imbalance calculation."""
        ob = sample_order_book

        imbalance = ob.imbalance(levels=2)
        bid_vol = 1.5 + 2.0
        ask_vol = 1.0 + 2.5
        expected = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        assert abs(imbalance - expected) < 1e-10

    def test_empty_order_book(self, empty_order_book):
        """Test empty order book edge cases."""
        ob = empty_order_book

        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.mid_price is None
        assert ob.spread is None

    def test_deep_order_book_imbalance(self, deep_order_book):
        """Test imbalance with deep order book."""
        ob = deep_order_book

        # Imbalance at different levels
        imb_5 = ob.imbalance(levels=5)
        imb_10 = ob.imbalance(levels=10)

        assert isinstance(imb_5, float)
        assert -1 <= imb_5 <= 1
        assert -1 <= imb_10 <= 1


class TestOptionContract:
    """Test OptionContract dataclass."""

    def test_option_contract_creation(self, sample_option_contract):
        """Test option contract creation (coin-margined)."""
        contract = sample_option_contract

        assert contract.underlying == "BTC-USD"
        assert contract.strike == 50000
        assert contract.option_type == OptionType.CALL
        assert contract.inverse is True
        assert contract.is_coin_margined is True

    def test_instrument_name(self, sample_option_contract):
        """Test instrument name generation."""
        contract = sample_option_contract
        name = contract.instrument_name

        assert "BTC-USD" in name
        assert "50000" in name
        assert name.endswith("-C")

    def test_time_to_expiry(self, sample_option_contract):
        """Test time to expiry calculation."""
        contract = sample_option_contract
        as_of = datetime(2024, 6, 1)

        tte = contract.time_to_expiry(as_of)
        assert tte > 0
        assert tte < 1  # Less than 1 year

    def test_expired_contract(self, sample_option_contract):
        """Test expired contract."""
        contract = sample_option_contract
        as_of = datetime(2025, 1, 1)

        tte = contract.time_to_expiry(as_of)
        assert tte == 0


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self, sample_position):
        """Test position creation."""
        pos = sample_position

        assert pos.instrument == "BTC-PERPETUAL"
        assert pos.size == 1.5
        assert pos.avg_entry_price == 49500

    def test_unrealized_pnl(self, sample_position):
        """Test unrealized PnL calculation."""
        pos = sample_position

        # Mark at 51000
        pnl = pos.unrealized_pnl(51000)
        expected = 1.5 * (51000 - 49500)
        assert pnl == expected

    def test_apply_fill_buy(self, sample_position):
        """Test applying buy fill."""
        pos = sample_position

        fill = Fill(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            side=OrderSide.BUY,
            price=50000,
            size=0.5
        )

        new_pos = pos.apply_fill(fill)

        assert new_pos.size == 2.0
        # Average price should be weighted
        expected_avg = (1.5 * 49500 + 0.5 * 50000) / 2.0
        assert abs(new_pos.avg_entry_price - expected_avg) < 1

    def test_apply_fill_sell(self, sample_position):
        """Test applying sell fill."""
        pos = sample_position

        fill = Fill(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            side=OrderSide.SELL,
            price=51000,
            size=1.0
        )

        new_pos = pos.apply_fill(fill)

        assert new_pos.size == 0.5
        assert new_pos.avg_entry_price == pos.avg_entry_price  # Unchanged

    def test_apply_fill_short(self):
        """Test applying fill that creates short position."""
        pos = Position(instrument="BTC-PERPETUAL", size=1.0, avg_entry_price=50000)

        fill = Fill(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            side=OrderSide.SELL,
            price=51000,
            size=2.0
        )

        new_pos = pos.apply_fill(fill)

        assert new_pos.size == -1.0
        assert new_pos.avg_entry_price == 51000

    def test_apply_fill_flatten_position(self):
        """Test applying fill that fully closes position resets average."""
        pos = Position(instrument="BTC-PERPETUAL", size=1.0, avg_entry_price=50000)

        fill = Fill(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            side=OrderSide.SELL,
            price=50500,
            size=1.0
        )

        new_pos = pos.apply_fill(fill)

        assert new_pos.size == 0.0
        assert new_pos.avg_entry_price == 0.0

    def test_apply_fill_wrong_instrument(self, sample_position):
        """Test error on wrong instrument fill."""
        pos = sample_position

        fill = Fill(
            timestamp=datetime.now(),
            instrument="ETH-PERPETUAL",
            side=OrderSide.BUY,
            price=3000,
            size=1.0
        )

        with pytest.raises(ValueError, match="Instrument mismatch"):
            pos.apply_fill(fill)


class TestGreeks:
    """Test Greeks dataclass."""

    def test_greeks_creation(self, sample_greeks):
        """Test Greeks creation."""
        g = sample_greeks

        assert g.delta == 0.5
        assert g.gamma == 0.001
        assert g.theta == -10.0
        assert g.vega == 5.0

    def test_optional_greeks(self):
        """Test Greeks with optional fields."""
        g = Greeks(delta=0.5, gamma=0.001, theta=-5.0, vega=2.0)

        assert g.rho is None
        assert g.vanna is None


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test trade creation."""
        trade = Trade(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            price=50000,
            size=1.5,
            side=OrderSide.BUY,
            trade_id="abc123"
        )

        assert trade.price == 50000
        assert trade.size == 1.5
        assert trade.side == OrderSide.BUY
        assert trade.trade_id == "abc123"

    def test_trade_without_id(self):
        """Test trade without optional ID."""
        trade = Trade(
            timestamp=datetime.now(),
            instrument="BTC-PERPETUAL",
            price=50000,
            size=1.5,
            side=OrderSide.SELL
        )

        assert trade.trade_id is None
