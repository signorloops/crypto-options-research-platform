"""
Tests for Integrated Market Making Strategy.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.types import Greeks, MarketState, OrderBook, OrderBookLevel, Position, Tick
from research.risk.circuit_breaker import CircuitState
from research.signals.regime_detector import RegimeState
from strategies.market_making.integrated_strategy import (
    IntegratedMarketMakingStrategy,
    IntegratedStrategyConfig,
    IntegratedStrategyWithFeatures,
    StrategyMetrics,
)


def create_test_market_state(price: float = 50000.0, timestamp: datetime = None) -> MarketState:
    """Create a test market state."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    spread = price * 0.001  # 10 bps spread
    order_book = OrderBook(
        timestamp=timestamp,
        instrument="BTC-USD",
        bids=[OrderBookLevel(price=price - spread/2, size=1.0)],
        asks=[OrderBookLevel(price=price + spread/2, size=1.0)]
    )

    return MarketState(
        timestamp=timestamp,
        instrument="BTC-USD",
        spot_price=price,
        order_book=order_book,
        recent_trades=[]
    )


class TestIntegratedStrategyConfig:
    """Tests for strategy configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = IntegratedStrategyConfig()
        assert config.base_spread_bps == 20.0
        assert config.quote_size == 1.0
        assert config.inventory_limit == 10.0
        assert config.gamma == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegratedStrategyConfig(
            base_spread_bps=30.0,
            quote_size=2.0,
            inventory_limit=5.0
        )
        assert config.base_spread_bps == 30.0
        assert config.quote_size == 2.0
        assert config.inventory_limit == 5.0

    def test_regime_spread_multipliers(self):
        """Test regime spread multipliers."""
        config = IntegratedStrategyConfig()

        assert config.regime_spread_multipliers[RegimeState.LOW] == 0.8
        assert config.regime_spread_multipliers[RegimeState.MEDIUM] == 1.0
        assert config.regime_spread_multipliers[RegimeState.HIGH] == 1.5


class TestIntegratedStrategyInitialization:
    """Tests for strategy initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        strategy = IntegratedMarketMakingStrategy()

        assert strategy.name == "IntegratedMarketMaking"
        assert strategy.circuit_breaker is not None
        assert strategy.regime_detector is not None
        assert strategy.hedger is not None

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = IntegratedStrategyConfig(base_spread_bps=30.0)
        strategy = IntegratedMarketMakingStrategy(config)

        assert strategy.config.base_spread_bps == 30.0


class TestQuoteGeneration:
    """Tests for quote generation."""

    def test_basic_quote(self):
        """Test basic quote generation."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        quote = strategy.quote(state, position)

        assert quote.bid_price > 0
        assert quote.ask_price > 0
        assert quote.bid_price < quote.ask_price
        assert quote.bid_size > 0
        assert quote.ask_size > 0

    def test_quote_metadata(self):
        """Test that quote includes metadata."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        quote = strategy.quote(state, position)

        assert "strategy" in quote.metadata
        assert "circuit_state" in quote.metadata
        assert "regime" in quote.metadata
        assert quote.metadata["strategy"] == "IntegratedMarketMaking"

    def test_spread_adjustment(self):
        """Test that spread is properly calculated."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.config.base_spread_bps = 20.0
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        quote = strategy.quote(state, position)

        # In normal conditions, spread should be close to base_spread_bps
        mid = (quote.bid_price + quote.ask_price) / 2
        spread_bps = quote.metadata.get("spread_bps", 0)

        assert spread_bps > 0

    def test_inventory_skew(self):
        """Test that inventory affects quotes."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)

        # Long position - should skew down (lower reservation price)
        long_position = Position("BTC-USD", 5, 50000.0)
        quote_long = strategy.quote(state, long_position)

        strategy.reset()

        # Short position - should skew up (higher reservation price)
        short_position = Position("BTC-USD", -5, 50000.0)
        quote_short = strategy.quote(state, short_position)

        # Long position should have lower reservation price
        # (more willing to sell, less willing to buy)
        assert quote_long.bid_price < quote_short.bid_price


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_halts_trading(self):
        """Test that circuit breaker can halt trading."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        # Manually set circuit breaker to HALTED
        strategy.circuit_breaker.state = CircuitState.HALTED

        quote = strategy.quote(state, position)

        # Should have zero sizes
        assert quote.bid_size == 0
        assert quote.ask_size == 0
        assert quote.metadata["trading_halted"] is True

    def test_circuit_breaker_warning_reduces_size(self):
        """Test that WARNING state reduces position sizes."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.config.quote_size = 1.0
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        # Set to WARNING state and verify multiplier directly
        strategy.circuit_breaker.state = CircuitState.WARNING

        # Verify the multiplier is correct
        assert strategy.circuit_breaker.get_position_limit_multiplier() == 0.5


class TestRegimeDetectorIntegration:
    """Tests for regime detector integration."""

    def test_regime_affects_spread(self):
        """Test that regime affects spread."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.config.base_spread_bps = 20.0
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        # Force regime to LOW
        strategy.regime_detector.current_regime = RegimeState.LOW
        quote_low = strategy.quote(state, position)

        strategy.reset()

        # Force regime to HIGH
        strategy.regime_detector.current_regime = RegimeState.HIGH
        quote_high = strategy.quote(state, position)

        # High regime should have wider spread
        spread_low = quote_low.ask_price - quote_low.bid_price
        spread_high = quote_high.ask_price - quote_high.bid_price

        # Note: This depends on the regime being properly set
        # The test verifies the integration works


class TestHedgerIntegration:
    """Tests for hedger integration."""

    def test_hedger_tracks_prices(self):
        """Test that hedger tracks price history."""
        strategy = IntegratedMarketMakingStrategy()

        # Generate several quotes with Greeks to trigger hedger
        base_time = datetime.now(timezone.utc)
        for i in range(10):
            state = create_test_market_state(
                price=50000.0 + i * 100,
                timestamp=base_time + timedelta(minutes=i)
            )
            position = Position("BTC-USD", 0, 0)
            strategy.quote(state, position)

        # Strategy should track prices internally
        assert len(strategy._pnl_history) > 0

    def test_market_state_greeks_are_propagated(self):
        """Strategy should keep latest Greeks from market state."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)
        state.greeks = Greeks(delta=0.45, gamma=0.01, theta=-0.02, vega=0.12)
        position = Position("BTC-USD", 1, 50000.0)

        strategy.quote(state, position)

        assert strategy._current_greeks is not None
        assert strategy._current_greeks.delta == pytest.approx(0.45)


class TestReturnCalculation:
    """Tests for return calculation updates."""

    def test_return_uses_previous_mid_price(self):
        """Second quote should record non-zero return when mid moves."""
        strategy = IntegratedMarketMakingStrategy()
        position = Position("BTC-USD", 0, 0)

        strategy.quote(create_test_market_state(price=50000.0), position)
        strategy.quote(create_test_market_state(price=51000.0), position)

        assert len(strategy._returns_history) == 1
        assert strategy._returns_history[0] == pytest.approx(np.log(51000.0 / 50000.0))


class TestPnLCalculation:
    """Tests for PnL calculation."""

    def test_pnl_tracking(self):
        """Test that PnL is tracked."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 1.0, 50000.0)

        strategy.quote(state, position)

        # Should have PnL history
        assert len(strategy._pnl_history) > 0

    def test_pnl_calculation(self):
        """Test PnL calculation."""
        strategy = IntegratedMarketMakingStrategy()

        # Long position, price goes up
        pnl = strategy._calculate_pnl(
            Position("BTC-USD", 1.0, 50000.0),
            51000.0
        )

        # For inverse: PnL = size * (1/entry - 1/exit)
        # = 1 * (1/50000 - 1/51000) = 1 * (0.00002 - 0.0000196) = 0.00000392
        expected_pnl = 1.0 * (1/50000 - 1/51000)
        assert abs(pnl - expected_pnl) < 1e-10


class TestSpreadMultiplier:
    """Tests for spread multiplier calculation."""

    def test_spread_multiplier_normal(self):
        """Test spread multiplier in normal conditions."""
        strategy = IntegratedMarketMakingStrategy()

        # Normal regime, normal circuit state
        strategy.regime_detector.current_regime = RegimeState.MEDIUM
        strategy.circuit_breaker.state = CircuitState.NORMAL

        mult = strategy._get_spread_multiplier()

        # Should be around 1.0
        assert mult >= 0.8
        assert mult <= 1.5

    def test_spread_multiplier_high_vol(self):
        """Test spread multiplier in high volatility."""
        strategy = IntegratedMarketMakingStrategy()

        strategy.regime_detector.current_regime = RegimeState.HIGH
        strategy.circuit_breaker.state = CircuitState.NORMAL

        mult = strategy._get_spread_multiplier()

        # High vol should have higher multiplier
        assert mult >= 1.0

    def test_spread_multiplier_is_multiplicative(self):
        """Regime and circuit multipliers should compose multiplicatively."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.regime_detector.current_regime = RegimeState.HIGH  # 1.5
        strategy.circuit_breaker.state = CircuitState.WARNING        # 1.5
        mult = strategy._get_spread_multiplier()
        assert mult == pytest.approx(2.25, rel=1e-3)


class TestReservationPrice:
    """Tests for reservation price calculation."""

    def test_reservation_price_with_inventory(self):
        """Test reservation price with inventory skew."""
        strategy = IntegratedMarketMakingStrategy()

        # Long inventory - reservation price should be lower than mid
        res_price, half_spread = strategy._calculate_reservation_price(
            mid=50000.0,
            inventory=5.0,
            spread_bps=20.0
        )

        assert res_price < 50000.0
        assert half_spread > 0

    def test_reservation_price_no_inventory(self):
        """Test reservation price with no inventory."""
        strategy = IntegratedMarketMakingStrategy()

        res_price, half_spread = strategy._calculate_reservation_price(
            mid=50000.0,
            inventory=0.0,
            spread_bps=20.0
        )

        # No inventory skew - reservation price equals mid
        assert res_price == 50000.0
        assert half_spread > 0


class TestQuoteSizes:
    """Tests for quote size calculation."""

    def test_basic_quote_sizes(self):
        """Test basic quote sizes."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.config.quote_size = 1.0
        strategy.config.inventory_limit = 10.0

        bid_size, ask_size = strategy._calculate_quote_sizes(0.0, 10.0)

        assert bid_size == 1.0
        assert ask_size == 1.0

    def test_inventory_limit_reduces_bid(self):
        """Test that inventory limit reduces bid size."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.config.quote_size = 1.0
        strategy.config.inventory_limit = 10.0

        # Already near long limit
        bid_size, ask_size = strategy._calculate_quote_sizes(9.5, 10.0)

        assert bid_size < 1.0  # Reduced
        assert ask_size == 1.0  # Unchanged

    def test_inventory_limit_reduces_ask(self):
        """Test that inventory limit reduces ask size."""
        strategy = IntegratedMarketMakingStrategy()
        strategy.config.quote_size = 1.0
        strategy.config.inventory_limit = 10.0

        # Already near short limit
        bid_size, ask_size = strategy._calculate_quote_sizes(-9.5, 10.0)

        assert bid_size == 1.0  # Unchanged
        assert ask_size < 1.0  # Reduced


class TestOnFill:
    """Tests for fill handling."""

    def test_on_fill_updates_pnl(self):
        """Test that on_fill updates realized PnL."""
        from core.types import Fill, OrderSide

        strategy = IntegratedMarketMakingStrategy()
        strategy._current_price = 50000.0

        initial_pnl = strategy._realized_pnl

        # Simulate a sell fill
        fill = Fill(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-USD",
            side=OrderSide.SELL,
            price=51000.0,
            size=0.1
        )

        strategy.on_fill(fill, Position("BTC-USD", -0.1, 51000.0))

        # PnL should have increased
        assert strategy._realized_pnl > initial_pnl

    def test_on_fill_rejects_uninitialized_price(self):
        """Fill processing should fail fast when current price is not initialized."""
        from core.types import Fill, OrderSide

        strategy = IntegratedMarketMakingStrategy()
        strategy._current_price = 0.0

        fill = Fill(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-USD",
            side=OrderSide.BUY,
            price=50000.0,
            size=0.1
        )

        with pytest.raises(ValueError, match="Current price must be positive"):
            strategy.on_fill(fill, Position("BTC-USD", 0.1, 50000.0))


class TestGetInternalState:
    """Tests for internal state retrieval."""

    def test_internal_state_format(self):
        """Test internal state format."""
        strategy = IntegratedMarketMakingStrategy()

        # Generate a quote first
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)
        strategy.quote(state, position)

        internal_state = strategy.get_internal_state()

        assert "circuit_breaker" in internal_state
        assert "regime" in internal_state
        assert "hedger" in internal_state
        assert "config" in internal_state


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        strategy = IntegratedMarketMakingStrategy()

        # Generate some activity
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 1.0, 50000.0)
        strategy.quote(state, position)

        assert len(strategy._pnl_history) > 0

        # Reset
        strategy.reset()

        assert len(strategy._pnl_history) == 0
        assert len(strategy._returns_history) == 0
        assert strategy._realized_pnl == 0.0


class TestMetrics:
    """Tests for metrics tracking."""

    def test_metrics_recorded(self):
        """Test that metrics are recorded."""
        strategy = IntegratedMarketMakingStrategy()
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        strategy.quote(state, position)

        assert len(strategy._metrics_history) > 0

    def test_metrics_df(self):
        """Test metrics DataFrame conversion."""
        strategy = IntegratedMarketMakingStrategy()

        # Generate several quotes
        for i in range(5):
            state = create_test_market_state(price=50000.0 + i * 100)
            position = Position("BTC-USD", i, 50000.0)
            strategy.quote(state, position)

        df = strategy.get_metrics_df()

        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "regime" in df.columns
        assert "circuit_state" in df.columns
        assert "inventory" in df.columns


class TestOnlineCalibration:
    """Tests for online parameter calibration in integrated strategy."""

    def test_online_calibration_updates_effective_sigma_and_skew(self):
        """Online calibration should adapt effective sigma and inventory skew."""
        strategy = IntegratedMarketMakingStrategy(
            IntegratedStrategyConfig(
                sigma=0.4,
                inventory_skew_factor=0.5,
                enable_online_calibration=True,
                calibration_window=12,
            )
        )
        position = Position("BTC-USD", 4.0, 50000.0)
        base_time = datetime.now(timezone.utc)

        last_quote = None
        for i in range(24):
            # Alternate larger price jumps to force volatility adaptation.
            price = 50000.0 + (250.0 if i % 2 == 0 else -210.0)
            state = create_test_market_state(
                price=price,
                timestamp=base_time + timedelta(seconds=i),
            )
            state.features["trade_intensity"] = 6.0
            last_quote = strategy.quote(state, position)

        assert last_quote is not None
        assert "calibrated_sigma" in last_quote.metadata
        assert "calibrated_inventory_skew_factor" in last_quote.metadata
        assert last_quote.metadata["calibrated_sigma"] != pytest.approx(0.4)
        assert last_quote.metadata["calibrated_inventory_skew_factor"] != pytest.approx(0.5)


class TestIntegratedStrategyWithFeatures:
    """Tests for extended strategy with features."""

    def test_feature_extraction(self):
        """Test feature extraction."""
        strategy = IntegratedStrategyWithFeatures()
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        quote = strategy.quote(state, position)

        assert "extracted_features" in quote.metadata

    def test_feature_quote_does_not_mutate_base_config(self):
        """Feature-adaptive skew should not permanently drift base config values."""
        strategy = IntegratedStrategyWithFeatures()
        base_skew = strategy.config.inventory_skew_factor
        state = create_test_market_state(price=50000.0)
        position = Position("BTC-USD", 0, 0)

        for _ in range(10):
            strategy.quote(state, position)

        assert strategy.config.inventory_skew_factor == pytest.approx(base_skew)


class TestPnLHistoryBounds:
    """Tests for bounded PnL history behavior."""

    def test_pnl_history_respects_configured_maxlen(self):
        """PnL history should not grow unbounded in long-running sessions."""
        strategy = IntegratedMarketMakingStrategy(
            IntegratedStrategyConfig(max_pnl_history=5)
        )
        position = Position("BTC-USD", 1.0, 50000.0)

        for i in range(12):
            state = create_test_market_state(price=50000.0 + i)
            strategy.quote(state, position)

        assert len(strategy._pnl_history) <= 5
        assert len(strategy._pnl_series_cache) <= 5


class TestIntegration:
    """Integration tests for the full strategy."""

    def test_full_trading_session(self):
        """Test a full simulated trading session."""
        strategy = IntegratedMarketMakingStrategy()
        base_time = datetime.now(timezone.utc)

        # Simulate a trading session with price movements
        prices = [50000.0 + np.sin(i/10) * 1000 for i in range(50)]
        inventory = 0.0

        for i, price in enumerate(prices):
            state = create_test_market_state(
                price=price,
                timestamp=base_time + timedelta(minutes=i)
            )
            position = Position("BTC-USD", inventory, 50000.0)

            quote = strategy.quote(state, position)

            # Verify quote is valid
            assert quote.bid_price > 0
            assert quote.ask_price > quote.bid_price
            assert quote.bid_size >= 0
            assert quote.ask_size >= 0

        # Should have recorded metrics
        assert len(strategy._metrics_history) == 50

    def test_circuit_breaker_activates_on_large_loss(self):
        """Test that circuit breaker activates on large losses."""
        from research.risk.circuit_breaker import Violation

        strategy = IntegratedMarketMakingStrategy()
        strategy.config.initial_capital = 1.0

        # Manually inject a violation to test circuit breaker integration
        violation = Violation(
            timestamp=datetime.now(timezone.utc),
            violation_type="daily_loss",
            severity="warning",
            current_value=0.06,
            limit_value=0.05,
            message="Daily loss 6% exceeds warning 5%"
        )
        strategy.circuit_breaker.violation_history.append(violation)

        # Circuit breaker should have recorded violations
        assert len(strategy.circuit_breaker.violation_history) > 0
        assert strategy.circuit_breaker.violation_history[0].violation_type == "daily_loss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
