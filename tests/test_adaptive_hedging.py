"""
Tests for Adaptive Delta Hedger.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.types import Greeks
from research.hedging.adaptive_delta import (
    AdaptiveDeltaHedger,
    AdaptiveHedgeConfig,
    HedgeDecision,
    SimpleDeltaHedger,
)


class TestAdaptiveHedgeConfig:
    """Tests for AdaptiveHedgeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AdaptiveHedgeConfig()
        assert config.base_hedge_interval_minutes == 30
        assert config.price_drop_threshold_pct == 0.05
        assert config.hedge_frequency_multiplier == 1.5
        assert config.gamma_threshold == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = AdaptiveHedgeConfig(
            base_hedge_interval_minutes=60,
            price_drop_threshold_pct=0.03,
            hedge_frequency_multiplier=2.0
        )
        assert config.base_hedge_interval_minutes == 60
        assert config.price_drop_threshold_pct == 0.03
        assert config.hedge_frequency_multiplier == 2.0


class TestAdaptiveDeltaHedgerInitialization:
    """Tests for hedger initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        hedger = AdaptiveDeltaHedger()
        assert hedger.last_hedge_time is None
        assert len(hedger.price_history) == 0
        assert len(hedger.hedge_history) == 0

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = AdaptiveHedgeConfig(base_hedge_interval_minutes=15)
        hedger = AdaptiveDeltaHedger(config)
        assert hedger.config.base_hedge_interval_minutes == 15


class TestPriceHistory:
    """Tests for price history tracking."""

    def test_update_price(self):
        """Test price update."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        hedger.update_price(now, 50000.0)
        assert len(hedger.price_history) == 1

        hedger.update_price(now + timedelta(minutes=1), 51000.0)
        assert len(hedger.price_history) == 2

    def test_price_history_maxlen(self):
        """Test price history maximum length."""
        config = AdaptiveHedgeConfig(price_history_window=5)
        hedger = AdaptiveDeltaHedger(config)
        now = datetime.now(timezone.utc)

        for i in range(10):
            hedger.update_price(now + timedelta(minutes=i), 50000.0 + i)

        assert len(hedger.price_history) == 5


class TestShouldHedge:
    """Tests for hedge decision logic."""

    def test_first_hedge(self):
        """Test that first hedge is triggered."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.5, gamma=0.01, theta=-0.1, vega=0.2)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        assert decision.should_hedge is True
        assert "First" in decision.reason or "Time" in decision.reason

    def test_time_based_hedge(self):
        """Test time-based hedge trigger."""
        config = AdaptiveHedgeConfig(base_hedge_interval_minutes=30)
        hedger = AdaptiveDeltaHedger(config)
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.1, gamma=0.01, theta=-0.1, vega=0.2)

        # Set last hedge time
        hedger.last_hedge_time = now - timedelta(minutes=35)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        assert decision.should_hedge is True

    def test_no_hedge_before_interval(self):
        """Test that hedge is not triggered before interval."""
        config = AdaptiveHedgeConfig(base_hedge_interval_minutes=30)
        hedger = AdaptiveDeltaHedger(config)
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.01, gamma=0.001, theta=-0.01, vega=0.02)

        # Set last hedge time to recent
        hedger.last_hedge_time = now - timedelta(minutes=5)

        # Add some price history
        for i in range(10):
            hedger.update_price(now - timedelta(minutes=10-i), 50000.0)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        assert decision.should_hedge is False

    def test_delta_deviation_trigger(self):
        """Test that large delta deviation triggers hedge."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        # Large delta
        greeks = Greeks(delta=0.5, gamma=0.01, theta=-0.1, vega=0.2)

        # Set last hedge time to recent
        hedger.last_hedge_time = now - timedelta(minutes=5)

        # Add price history
        for i in range(10):
            hedger.update_price(now - timedelta(minutes=10-i), 50000.0)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        # Should hedge due to large delta deviation
        assert decision.should_hedge is True
        assert decision.current_delta == 0.5

    def test_price_drop_acceleration(self):
        """Test that price drop accelerates hedging."""
        config = AdaptiveHedgeConfig(
            base_hedge_interval_minutes=30,
            price_drop_threshold_pct=0.05
        )
        hedger = AdaptiveDeltaHedger(config)
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.1, gamma=0.01, theta=-0.1, vega=0.2)

        # Set last hedge time (20 minutes ago, less than 30 min interval)
        hedger.last_hedge_time = now - timedelta(minutes=20)

        # Add price history with significant drop
        hedger.update_price(now - timedelta(minutes=20), 50000.0)
        hedger.update_price(now - timedelta(minutes=15), 52000.0)  # Peak
        hedger.update_price(now - timedelta(minutes=10), 51000.0)
        hedger.update_price(now - timedelta(minutes=5), 49000.0)   # 5.8% drop from peak

        decision = hedger.should_hedge(now, 49000.0, greeks, 1.0)

        # Should hedge due to price drop acceleration
        assert decision.should_hedge is True
        assert decision.urgency in ["high", "critical", "normal"]

    def test_zero_delta_does_not_trigger_zero_size_hedge(self):
        """High urgency with zero delta should not return should_hedge=True with zero size."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)
        hedger.last_hedge_time = now - timedelta(minutes=1)

        # Build history that creates critical urgency via sharp drop.
        hedger.update_price(now - timedelta(minutes=4), 55000.0)
        hedger.update_price(now - timedelta(minutes=3), 56000.0)
        hedger.update_price(now - timedelta(minutes=2), 54000.0)

        # Zero net delta exposure.
        greeks = Greeks(delta=0.0, gamma=0.03, theta=-0.1, vega=0.2)
        decision = hedger.should_hedge(now, 50000.0, greeks, position_size=1.0)

        assert decision.hedge_size == 0.0
        assert decision.should_hedge is False


class TestHedgeSizeCalculation:
    """Tests for hedge size calculation."""

    def test_basic_hedge_size(self):
        """Test basic hedge size calculation."""
        hedger = AdaptiveDeltaHedger()

        size = hedger._calculate_hedge_size(0.5, 0.0, 50000.0, 0.01)

        assert size < 0  # Negative because we need to sell to reduce delta
        assert abs(size) > 0

    def test_hedge_size_respects_limits(self):
        """Test that hedge size respects min/max limits."""
        config = AdaptiveHedgeConfig(min_hedge_size=0.0001, max_hedge_size_pct=0.5)
        hedger = AdaptiveDeltaHedger(config)

        # Small delta difference
        size = hedger._calculate_hedge_size(0.001, 0.0, 50000.0, 0.01)

        # Should respect min_hedge_size
        assert abs(size) >= config.min_hedge_size

        # Large delta difference
        size = hedger._calculate_hedge_size(1.0, 0.0, 50000.0, 0.01)

        # Should respect max_hedge_size_pct
        assert abs(size) <= 1.0 * config.max_hedge_size_pct

    def test_inverse_adjustment(self):
        """Test coin-margined adjustment."""
        config = AdaptiveHedgeConfig(inverse=True)
        hedger = AdaptiveDeltaHedger(config)

        # At lower price, hedge size should be larger
        size_low_price = hedger._calculate_hedge_size(0.5, 0.0, 30000.0, 0.01)
        size_high_price = hedger._calculate_hedge_size(0.5, 0.0, 60000.0, 0.01)

        # Lower price should result in larger hedge size (inverse effect)
        # The factor is 50000/price, so at 30000 it's 1.67x, at 60000 it's 0.83x
        assert abs(size_low_price) >= abs(size_high_price)


class TestUrgency:
    """Tests for urgency determination."""

    def test_critical_urgency(self):
        """Test critical urgency detection."""
        hedger = AdaptiveDeltaHedger()

        urgency = hedger._determine_urgency(0.10, 0.0, 0.01, 0.1)
        assert urgency == "critical"

    def test_high_urgency_price_drop(self):
        """Test high urgency on price drop."""
        hedger = AdaptiveDeltaHedger()

        urgency = hedger._determine_urgency(0.05, 0.0, 0.01, 0.05)
        assert urgency == "high"

    def test_high_urgency_delta_deviation(self):
        """Test high urgency on delta deviation."""
        hedger = AdaptiveDeltaHedger()

        urgency = hedger._determine_urgency(0.02, 0.0, 0.01, 0.2)
        assert urgency == "high"

    def test_low_urgency(self):
        """Test low urgency in calm conditions."""
        hedger = AdaptiveDeltaHedger()

        urgency = hedger._determine_urgency(0.01, 0.01, 0.005, 0.02)
        assert urgency == "low"


class TestExecuteHedge:
    """Tests for hedge execution."""

    def test_execute_hedge(self):
        """Test hedge execution recording."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        hedger.execute_hedge(now, -0.5, 50000.0)

        assert hedger.last_hedge_time == now
        assert len(hedger.hedge_history) == 1
        assert hedger.hedge_history[0][1] == -0.5

    def test_multiple_hedges(self):
        """Test multiple hedge executions."""
        hedger = AdaptiveDeltaHedger()
        base_time = datetime.now(timezone.utc)

        for i in range(5):
            hedger.execute_hedge(
                base_time + timedelta(minutes=i*10),
                -0.1 * (i + 1),
                50000.0 + i * 100
            )

        assert len(hedger.hedge_history) == 5


class TestHedgeStats:
    """Tests for hedge statistics."""

    def test_stats_empty(self):
        """Test stats with no hedges."""
        hedger = AdaptiveDeltaHedger()

        stats = hedger.get_hedge_stats()

        assert stats["total_hedges"] == 0
        assert stats["avg_hedge_size"] == 0.0
        assert stats["last_hedge_time"] is None

    def test_stats_with_hedges(self):
        """Test stats with hedges."""
        hedger = AdaptiveDeltaHedger()
        base_time = datetime.now(timezone.utc)

        for i in range(5):
            hedger.execute_hedge(
                base_time + timedelta(minutes=i),
                -0.1 * (i + 1),
                50000.0
            )

        stats = hedger.get_hedge_stats()

        assert stats["total_hedges"] == 5
        assert stats["avg_hedge_size"] > 0
        assert stats["max_hedge_size"] >= stats["min_hedge_size"]
        assert stats["last_hedge_time"] is not None


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        # Add some data
        hedger.update_price(now, 50000.0)
        hedger.execute_hedge(now, -0.5, 50000.0)

        assert len(hedger.price_history) > 0
        assert len(hedger.hedge_history) > 0
        assert hedger.last_hedge_time is not None

        # Reset
        hedger.reset()

        assert len(hedger.price_history) == 0
        assert len(hedger.hedge_history) == 0
        assert hedger.last_hedge_time is None


class TestSimpleDeltaHedger:
    """Tests for simple delta hedger."""

    def test_simple_hedger_first_hedge(self):
        """Test simple hedger first hedge."""
        hedger = SimpleDeltaHedger(hedge_interval_minutes=30)
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.5, gamma=0.01, theta=-0.1, vega=0.2)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        assert decision.should_hedge is True
        assert decision.reason == "First hedge"

    def test_simple_hedger_time_based(self):
        """Test simple hedger time-based trigger."""
        hedger = SimpleDeltaHedger(hedge_interval_minutes=30)
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.1, gamma=0.01, theta=-0.1, vega=0.2)

        hedger.last_hedge_time = now - timedelta(minutes=35)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        assert decision.should_hedge is True
        assert "Time-based" in decision.reason

    def test_simple_hedger_no_hedge(self):
        """Test simple hedger no hedge before interval."""
        hedger = SimpleDeltaHedger(hedge_interval_minutes=30)
        now = datetime.now(timezone.utc)
        greeks = Greeks(delta=0.1, gamma=0.01, theta=-0.1, vega=0.2)

        hedger.last_hedge_time = now - timedelta(minutes=10)

        decision = hedger.should_hedge(now, 50000.0, greeks, 1.0)

        assert decision.should_hedge is False


class TestPriceChangeCalculation:
    """Tests for price change calculations."""

    def test_price_change_pct(self):
        """Test price change percentage calculation."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        # Add price history
        for i in range(10):
            hedger.update_price(now - timedelta(minutes=10-i), 50000.0)

        change = hedger._calculate_price_change_pct(51000.0)

        # 2% increase
        assert change == pytest.approx(0.02, abs=0.001)

    def test_price_drop_pct(self):
        """Test price drop percentage calculation."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        hedger.last_hedge_time = now - timedelta(minutes=20)

        # Add price history with peak
        hedger.update_price(now - timedelta(minutes=20), 50000.0)
        hedger.update_price(now - timedelta(minutes=15), 52000.0)  # Peak
        hedger.update_price(now - timedelta(minutes=10), 51000.0)

        drop = hedger._calculate_price_drop_pct(49000.0)

        # Drop from 52000 to 49000 = 5.77%
        assert drop == pytest.approx(0.0577, abs=0.001)

    def test_price_rise_pct(self):
        """Test price rise percentage calculation."""
        hedger = AdaptiveDeltaHedger()
        now = datetime.now(timezone.utc)

        hedger.last_hedge_time = now - timedelta(minutes=20)

        # Add price history with trough
        hedger.update_price(now - timedelta(minutes=20), 50000.0)
        hedger.update_price(now - timedelta(minutes=15), 48000.0)  # Trough
        hedger.update_price(now - timedelta(minutes=10), 49000.0)

        rise = hedger._calculate_price_rise_pct(51000.0)

        # Rise from 48000 to 51000 = 6.25%
        assert rise == pytest.approx(0.0625, abs=0.001)


class TestAdjustedInterval:
    """Tests for adjusted hedge interval calculation."""

    def test_base_interval_unchanged(self):
        """Test that base interval is unchanged in calm conditions."""
        hedger = AdaptiveDeltaHedger()
        base = timedelta(minutes=30)

        adjusted = hedger._calculate_adjusted_interval(base, 0.0, 0.0, 0.005)

        # Should be approximately the same
        assert adjusted.total_seconds() == pytest.approx(base.total_seconds(), rel=0.1)

    def test_price_drop_shortens_interval(self):
        """Test that price drop shortens hedge interval."""
        hedger = AdaptiveDeltaHedger()
        base = timedelta(minutes=30)

        adjusted = hedger._calculate_adjusted_interval(base, 0.10, 0.0, 0.01)

        # Should be shorter than base
        assert adjusted.total_seconds() < base.total_seconds()

    def test_high_gamma_shortens_interval(self):
        """Test that high gamma shortens hedge interval."""
        hedger = AdaptiveDeltaHedger()
        base = timedelta(minutes=30)

        adjusted = hedger._calculate_adjusted_interval(base, 0.0, 0.0, 0.05)

        # Should be shorter than base
        assert adjusted.total_seconds() < base.total_seconds()


class TestIntegration:
    """Integration tests for adaptive hedger."""

    def test_full_hedge_workflow(self):
        """Test complete hedge workflow."""
        config = AdaptiveHedgeConfig(base_hedge_interval_minutes=30)
        hedger = AdaptiveDeltaHedger(config)
        base_time = datetime.now(timezone.utc)

        # Initial hedge
        greeks = Greeks(delta=0.5, gamma=0.01, theta=-0.1, vega=0.2)
        decision = hedger.should_hedge(base_time, 50000.0, greeks, 1.0)

        assert decision.should_hedge is True
        hedger.execute_hedge(base_time, decision.hedge_size, 50000.0)

        # Price drops significantly after 20 minutes
        new_time = base_time + timedelta(minutes=20)
        for i in range(20):
            hedger.update_price(
                base_time + timedelta(minutes=i),
                50000.0 - i * 200  # Price dropping
            )

        greeks2 = Greeks(delta=0.6, gamma=0.015, theta=-0.12, vega=0.25)
        decision2 = hedger.should_hedge(new_time, 46000.0, greeks2, 1.0)

        # Should hedge due to price drop acceleration
        assert decision2.should_hedge is True
        assert decision2.urgency in ["high", "critical"]

        # Execute second hedge
        hedger.execute_hedge(new_time, decision2.hedge_size, 46000.0)

        # Check stats
        stats = hedger.get_hedge_stats()
        assert stats["total_hedges"] == 2
