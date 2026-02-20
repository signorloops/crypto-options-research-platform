"""
Tests for Circuit Breaker system.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from research.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    PortfolioState,
    TradeAction,
    Violation,
    calculate_drawdown,
)


class TestCalculateDrawdown:
    """Tests for drawdown calculation."""

    def test_empty_series(self):
        """Test drawdown with empty series."""
        series = pd.Series()
        max_dd, dd_series = calculate_drawdown(series)
        assert max_dd == 0.0
        assert len(dd_series) == 0

    def test_no_drawdown(self):
        """Test with always increasing PnL."""
        series = pd.Series([0, 1, 2, 3, 4, 5])
        max_dd, _ = calculate_drawdown(series)
        assert max_dd == 0.0

    def test_single_drawdown(self):
        """Test with single drawdown."""
        series = pd.Series([0, 5, 4, 3, 6, 7])
        max_dd, _ = calculate_drawdown(series)
        # Peak at 5, trough at 3, drawdown = (3-5)/5 = -0.4
        assert max_dd == pytest.approx(-0.4, abs=0.01)

    def test_multiple_drawdowns(self):
        """Test with multiple drawdowns."""
        series = pd.Series([0, 10, 8, 9, 5, 7, 3, 8])
        max_dd, dd_series = calculate_drawdown(series)
        # Largest drawdown: peak 10, trough 3, drawdown = (3-10)/10 = -0.7
        assert max_dd == pytest.approx(-0.7, abs=0.01)
        assert len(dd_series) == len(series)

    def test_negative_series_detects_drawdown(self):
        """Negative-only PnL should still report drawdown instead of 0."""
        series = pd.Series([-0.05, -0.10, -0.20])
        max_dd, _ = calculate_drawdown(series)
        assert max_dd < 0


class TestPortfolioState:
    """Tests for PortfolioState."""

    def test_empty_portfolio(self):
        """Test with empty portfolio."""
        state = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={},
            cash=1000.0,
            initial_capital=1000.0
        )
        assert state.total_value == 1000.0
        assert state.daily_pnl == 0.0
        assert state.max_drawdown == 0.0

    def test_daily_pnl_calculation(self):
        """Test daily PnL calculation."""
        today = datetime.now(timezone.utc).date()
        timestamps = [
            datetime.combine(today, datetime.min.time()) + timedelta(hours=i)
            for i in range(5)
        ]
        pnl_series = pd.Series([0, 1, 2, 1, 3], index=timestamps)

        state = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1000.0
        )

        # Daily PnL = 3 - 0 = 3
        assert state.daily_pnl == 3.0
        # Daily PnL % = 3/1000 = 0.003
        assert state.daily_pnl_pct == pytest.approx(0.003, abs=0.0001)

    def test_position_concentration(self):
        """Test position concentration calculation."""
        from core.types import Position

        state = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={
                "BTC": Position("BTC", 1.0, 50000.0),
                "ETH": Position("ETH", 0.5, 3000.0),
            },
            cash=1000.0,
            initial_capital=50000.0
        )

        instrument, concentration = state.get_position_concentration()
        # BTC value = 1.0 * 50000 = 50000
        # Total = 50000 + 1500 + 1000 = 52500
        # BTC concentration = 50000/52500 ≈ 0.952
        assert instrument == "BTC"
        assert concentration > 0.9

    def test_total_value_includes_position_notional(self):
        """Total value should include positions to avoid concentration denominator distortion."""
        from core.types import Position

        state = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={
                "BTC": Position("BTC", 1.0, 50000.0),
                "ETH": Position("ETH", 0.5, 3000.0),
            },
            cash=1000.0,
            initial_capital=50000.0
        )

        expected_total = 1000.0 + 50000.0 + 1500.0
        assert state.total_value == pytest.approx(expected_total)

    def test_single_position_does_not_trigger_concentration_metric(self):
        """Single-instrument portfolio should not be flagged by cross-position concentration."""
        from core.types import Position

        state = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={"BTC": Position("BTC", 1.0, 50000.0)},
            cash=1000.0,
            initial_capital=50000.0
        )

        instrument, concentration = state.get_position_concentration()
        assert instrument == ""
        assert concentration == 0.0


class TestCircuitBreakerInitialization:
    """Tests for circuit breaker initialization."""

    def test_default_config(self):
        """Test with default configuration."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.NORMAL
        assert cb.config.daily_loss_limit_pct == 0.10
        assert cb.config.max_drawdown_pct == 0.15
        assert len(cb.violation_history) == 0

    def test_custom_config(self):
        """Test with custom configuration."""
        config = CircuitBreakerConfig(
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.10
        )
        cb = CircuitBreaker(config)
        assert cb.config.daily_loss_limit_pct == 0.05
        assert cb.config.max_drawdown_pct == 0.10


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_normal_to_warning_daily_loss(self):
        """Test transition from NORMAL to WARNING on daily loss."""
        cb = CircuitBreaker()

        # Create portfolio with daily loss at warning threshold
        today = datetime.now(timezone.utc).date()
        timestamps = [datetime.combine(today, datetime.min.time()) + timedelta(minutes=i) for i in range(10)]
        pnl_series = pd.Series([0, -0.01, -0.02, -0.03, -0.04, -0.055], index=timestamps[:6])

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        # Daily loss is -0.055 which exceeds warning threshold of 0.05
        state = cb.check_risk_limits(portfolio)
        assert state == CircuitState.WARNING

    def test_normal_to_restricted_drawdown(self):
        """Test transition from NORMAL to RESTRICTED on drawdown."""
        cb = CircuitBreaker()

        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=10, freq='min')
        # Create drawdown exceeding 15% limit (critical severity)
        # Peak = 0.20, current = 0.0, drawdown = (0.0 - 0.20) / 0.20 = -1.0 (100%)
        # But we need at least 15% drawdown to trigger critical
        # Peak = 0.20, current = 0.17, drawdown = (0.17 - 0.20) / 0.20 = -0.15 (15%)
        pnl_series = pd.Series([0, 0.10, 0.20, 0.17], index=timestamps[:4])

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        state = cb.check_risk_limits(portfolio)
        # 15% drawdown is critical severity -> RESTRICTED
        assert state == CircuitState.RESTRICTED

    def test_warning_to_restricted_multiple_warnings(self):
        """Test transition from WARNING to RESTRICTED with multiple warnings."""
        cb = CircuitBreaker()
        cb.state = CircuitState.WARNING

        today = datetime.now(timezone.utc).date()
        timestamps = [datetime.combine(today, datetime.min.time()) + timedelta(minutes=i) for i in range(10)]

        # Create two warning-level violations
        pnl_series = pd.Series([0, -0.03, -0.06, -0.04, -0.055], index=timestamps[:5])

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        # Check multiple times to accumulate warnings
        for _ in range(3):
            state = cb.check_risk_limits(portfolio)

        assert state == CircuitState.RESTRICTED

    def test_restricted_to_halted(self):
        """Test transition from RESTRICTED to HALTED."""
        cb = CircuitBreaker()
        cb.state = CircuitState.RESTRICTED

        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=10, freq='min')
        # Severe drawdown while already restricted
        pnl_series = pd.Series([0, 0.05, 0.0, -0.10, -0.20], index=timestamps[:5])

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        state = cb.check_risk_limits(portfolio)
        assert state == CircuitState.HALTED

    def test_recovery_to_normal(self):
        """Test recovery from WARNING to NORMAL."""
        config = CircuitBreakerConfig(cooldown_period_seconds=0)  # No cooldown for testing
        cb = CircuitBreaker(config)
        cb.state = CircuitState.WARNING

        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=5, freq='min')
        # No violations - good PnL
        pnl_series = pd.Series([0, 0.01, 0.02, 0.03], index=timestamps[:4])

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        state = cb.check_risk_limits(portfolio)
        assert state == CircuitState.NORMAL


class TestCircuitBreakerCanTrade:
    """Tests for can_trade method."""

    def test_normal_state_allows_all_trades(self):
        """Test that NORMAL state allows all trade types."""
        cb = CircuitBreaker()
        cb.state = CircuitState.NORMAL

        for action in TradeAction:
            allowed, reason = cb.can_trade(action)
            assert allowed is True, f"{action.value} should be allowed in NORMAL state"

    def test_halted_state_blocks_all_trades(self):
        """Test that HALTED state blocks all trades."""
        cb = CircuitBreaker()
        cb.state = CircuitState.HALTED

        for action in TradeAction:
            allowed, reason = cb.can_trade(action)
            assert allowed is False, f"{action.value} should be blocked in HALTED state"
            assert "halted" in reason.lower()

    def test_restricted_state_allows_only_hedging(self):
        """Test that RESTRICTED state only allows hedging/liquidation."""
        cb = CircuitBreaker()
        cb.state = CircuitState.RESTRICTED

        # Should allow hedging and liquidation
        allowed, _ = cb.can_trade(TradeAction.HEDGING)
        assert allowed is True

        allowed, _ = cb.can_trade(TradeAction.LIQUIDATION)
        assert allowed is True

        # Should block market making and new positions
        allowed, reason = cb.can_trade(TradeAction.MARKET_MAKING)
        assert allowed is False
        assert "restricted" in reason.lower()

        allowed, reason = cb.can_trade(TradeAction.NEW_POSITION)
        assert allowed is False

    def test_warning_state_allows_with_message(self):
        """Test that WARNING state allows trades with warning message."""
        cb = CircuitBreaker()
        cb.state = CircuitState.WARNING

        allowed, reason = cb.can_trade(TradeAction.NEW_POSITION, size=1.0)
        assert allowed is True
        assert "warning" in reason.lower()


class TestCircuitBreakerMultipliers:
    """Tests for position limit and spread multipliers."""

    def test_position_limit_multipliers(self):
        """Test position limit multipliers for each state."""
        cb = CircuitBreaker()

        cb.state = CircuitState.NORMAL
        assert cb.get_position_limit_multiplier() == 1.0

        cb.state = CircuitState.WARNING
        assert cb.get_position_limit_multiplier() == 0.5

        cb.state = CircuitState.RESTRICTED
        assert cb.get_position_limit_multiplier() == 0.1

        cb.state = CircuitState.HALTED
        assert cb.get_position_limit_multiplier() == 0.0

    def test_spread_multipliers(self):
        """Test spread adjustment multipliers."""
        cb = CircuitBreaker()

        cb.state = CircuitState.NORMAL
        assert cb.get_spread_multiplier() == 1.0

        cb.state = CircuitState.WARNING
        assert cb.get_spread_multiplier() == 1.5

        cb.state = CircuitState.RESTRICTED
        assert cb.get_spread_multiplier() == 2.0

        cb.state = CircuitState.HALTED
        assert cb.get_spread_multiplier() == float('inf')


class TestCircuitBreakerCooldown:
    """Tests for cooldown functionality."""

    def test_cooldown_set_on_state_change(self):
        """Test that cooldown is set on state change."""
        config = CircuitBreakerConfig(cooldown_period_seconds=300)
        cb = CircuitBreaker(config)

        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=5, freq='min')
        pnl_series = pd.Series([0, -0.02, -0.04, -0.06, -0.08], index=timestamps)

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        cb.check_risk_limits(portfolio)

        assert cb.is_in_cooldown is True
        assert cb._cooldown_until is not None

    def test_cooldown_expires(self):
        """Test that cooldown expires after period."""
        config = CircuitBreakerConfig(cooldown_period_seconds=0)  # Immediate expiry
        cb = CircuitBreaker(config)
        cb._cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=1)

        assert cb.is_in_cooldown is False


class TestCircuitBreakerManualReset:
    """Tests for manual reset functionality."""

    def test_manual_reset_to_normal(self):
        """Test manual reset to NORMAL state."""
        cb = CircuitBreaker()
        cb.state = CircuitState.HALTED
        cb._consecutive_warnings = 5
        cb._warning_count = {"daily_loss": 3}

        cb.manual_reset("Test reset")

        assert cb.state == CircuitState.NORMAL
        assert cb._consecutive_warnings == 0


class TestCircuitBreakerVaRAndPerInstrument:
    """Tests for VaR integration and per-instrument limits."""

    def test_var_check_integrated_in_all_limits(self):
        config = CircuitBreakerConfig(
            var_95_limit_pct=0.0001,  # very tight to trigger
            var_method="hybrid",
            cooldown_period_seconds=0
        )
        cb = CircuitBreaker(config)

        idx = pd.date_range(start=datetime.now(timezone.utc) - timedelta(hours=80), periods=80, freq="h")
        # Synthetic volatile returns
        asset_returns = pd.DataFrame({
            "BTC": np.random.normal(0, 0.05, len(idx)),
            "ETH": np.random.normal(0, 0.06, len(idx)),
        }, index=idx)
        pnl_series = pd.Series(np.cumsum(np.random.normal(0, 0.02, len(idx))), index=idx)

        from core.types import Position
        portfolio = PortfolioState(
            timestamp=idx[-1].to_pydatetime(),
            positions={
                "BTC": Position("BTC", 1.0, 50000.0),
                "ETH": Position("ETH", 10.0, 3000.0),
            },
            cash=1000.0,
            pnl_series=pnl_series,
            asset_returns=asset_returns,
            initial_capital=100000.0
        )

        violations = cb._check_all_limits(portfolio)
        assert any(v.violation_type.startswith("var_95") or v.violation_type == "var_95" for v in violations)

    def test_per_instrument_notional_limit(self):
        config = CircuitBreakerConfig(
            enable_per_instrument_limits=True,
            per_instrument_notional_limit=1000.0,
            per_instrument_warning_notional=500.0
        )
        cb = CircuitBreaker(config)
        from core.types import Position
        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={"BTC": Position("BTC", 1.0, 50000.0)},
            cash=1000.0,
            initial_capital=100000.0
        )
        violations = cb._check_per_instrument_limits(portfolio)
        assert any(v.violation_type.startswith("instrument_notional:BTC") for v in violations)


class TestCircuitBreakerStateSeverityOrdering:
    """Regression tests for state severity comparisons."""

    def test_degradation_to_halted_triggers_alert(self, monkeypatch):
        """RESTRICTED -> HALTED must be treated as degradation."""
        cb = CircuitBreaker()
        cb.state = CircuitState.RESTRICTED

        called = {"alert": False}

        def fake_alert(state, violations):
            called["alert"] = True

        monkeypatch.setattr(cb, "_send_alert", fake_alert)
        violation = Violation(
            timestamp=datetime.now(timezone.utc),
            violation_type="drawdown",
            severity="critical",
            current_value=0.2,
            limit_value=0.15,
            message="critical drawdown",
        )

        cb._transition_state(CircuitState.HALTED, [violation])
        assert called["alert"] is True

    def test_improvement_resets_warning_counters(self):
        """HALTED -> RESTRICTED should reset warning counters."""
        cb = CircuitBreaker()
        cb.state = CircuitState.HALTED
        cb._consecutive_warnings = 3
        cb._warning_count = {"daily_loss": 2}

        cb._transition_state(CircuitState.RESTRICTED, [])

        assert cb._consecutive_warnings == 0
        assert cb._warning_count == {}


class TestCircuitBreakerStatus:
    """Tests for get_status method."""

    def test_status_format(self):
        """Test status dictionary format."""
        cb = CircuitBreaker()

        # Add a violation
        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=5, freq='min')
        pnl_series = pd.Series([0, -0.02, -0.04, -0.055], index=timestamps[:4])

        portfolio = PortfolioState(
            timestamp=timestamps[-1],
            positions={},
            cash=1000.0,
            pnl_series=pnl_series,
            initial_capital=1.0
        )

        cb.check_risk_limits(portfolio)

        status = cb.get_status()

        assert "state" in status
        assert "is_in_cooldown" in status
        assert "violation_count" in status
        assert "recent_violations" in status
        assert "position_limit_multiplier" in status
        assert "spread_multiplier" in status

        assert status["state"] == CircuitState.WARNING.value
        assert status["violation_count"] >= 1


class TestCircuitBreakerConcentration:
    """Tests for position concentration limits."""

    def test_concentration_warning(self):
        """Test warning on position concentration."""
        from core.types import Position

        cb = CircuitBreaker()

        state = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={
                "BTC": Position("BTC", 1.0, 25000.0),   # 25%
                "ETH": Position("ETH", 1.0, 25000.0),   # 25%
                "SOL": Position("SOL", 1.0, 25000.0),   # 25%
                "XRP": Position("XRP", 1.0, 25000.0),   # 25%
            },
            cash=1000.0,
            initial_capital=100000.0
        )

        # Largest concentration = 25%, should trigger warning (>=20%, <30%).
        violations = cb._check_all_limits(state)
        concentration_violations = [v for v in violations if v.violation_type == "concentration"]
        assert len(concentration_violations) > 0
        assert all(v.severity == "warning" for v in concentration_violations)

    def test_concentration_critical(self):
        """Test critical violation on extreme concentration."""
        from core.types import Position

        cb = CircuitBreaker()

        # Create extreme concentration with one dominant position.
        state = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            positions={
                "BTC": Position("BTC", 10.0, 50000.0),  # 500000 value
                "ETH": Position("ETH", 1.0, 20000.0),   # 20000 value
            },
            cash=1000.0,
            initial_capital=100000.0
        )

        violations = cb._check_all_limits(state)
        concentration_violations = [v for v in violations if v.violation_type == "concentration"]

        # Should trigger critical severity
        critical = [v for v in concentration_violations if v.severity == "critical"]
        assert len(critical) > 0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def test_full_scenario_simulation(self):
        """Test full scenario with multiple state changes."""
        config = CircuitBreakerConfig(cooldown_period_seconds=0)
        cb = CircuitBreaker(config)

        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=20, freq='min')

        # Simulate a trading session with various conditions
        # Note: State transitions depend on both current conditions and history
        # The state machine logic:
        # - critical_count >= 1 -> RESTRICTED
        # - warning_count >= 2 or (warning_count >= 1 and state == WARNING) -> RESTRICTED
        # - warning_count >= 1 -> WARNING
        # Note: daily_pnl_pct = daily_pnl / initial_capital
        # So for 5% warning threshold with initial_capital=1000, we need daily_pnl = -50
        scenarios = [
            # (pnl_series_values, expected_state, description)
            ([0.0], CircuitState.NORMAL, "Start normal"),
            ([0.0, 20.0], CircuitState.NORMAL, "Profitable"),
            ([0.0, -30.0], CircuitState.NORMAL, "Small loss (3%)"),
            ([0.0, -60.0], CircuitState.WARNING, "Daily loss warning (6% > 5%)"),
            ([0.0, -60.0, -60.0], CircuitState.RESTRICTED, "Same warning again -> RESTRICTED"),
        ]

        for i, (pnl_values, expected_state, desc) in enumerate(scenarios):
            pnl_series = pd.Series(pnl_values, index=[timestamps[0]] + [timestamps[j+1] for j in range(len(pnl_values)-1)])

            portfolio = PortfolioState(
                timestamp=timestamps[i],
                positions={},
                cash=1000.0 + pnl_values[-1],
                pnl_series=pnl_series,
                initial_capital=1000.0
            )

            state = cb.check_risk_limits(portfolio)
            assert state == expected_state, f"Step {i} ({desc}): expected {expected_state.value}, got {state.value}"

    def test_violation_history_accumulation(self):
        """Test that violations are properly recorded."""
        cb = CircuitBreaker()

        timestamps = pd.date_range(start=datetime.now(timezone.utc), periods=10, freq='min')

        for i in range(5):
            pnl_series = pd.Series([0, -0.06], index=[timestamps[0], timestamps[i]])

            portfolio = PortfolioState(
                timestamp=timestamps[i],
                positions={},
                cash=1000.0,
                pnl_series=pnl_series,
                initial_capital=1.0
            )

            cb.check_risk_limits(portfolio)

        # Should have accumulated violations
        assert len(cb.violation_history) > 0

        # Check violation structure
        violation = cb.violation_history[0]
        assert isinstance(violation, Violation)
        assert violation.timestamp is not None
        assert violation.violation_type in ["daily_loss", "drawdown", "concentration"]
        assert violation.severity in ["warning", "critical"]
