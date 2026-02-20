"""
Circuit Breaker system for risk management.

Implements a 4-tier circuit breaker system:
- NORMAL: Normal trading allowed
- WARNING: Reduced position sizing, tighter limits
- RESTRICTED: Only hedging/liquidation allowed
- HALTED: All trading suspended
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from core.types import Position, Portfolio
from research.risk.var import VaRCalculator, VaRResult


class CircuitState(Enum):
    """Circuit breaker states."""
    NORMAL = "normal"
    WARNING = "warning"
    RESTRICTED = "restricted"
    HALTED = "halted"


class TradeAction(Enum):
    """Types of trading actions."""
    MARKET_MAKING = "market_making"
    HEDGING = "hedging"
    LIQUIDATION = "liquidation"
    NEW_POSITION = "new_position"


@dataclass
class Violation:
    """Record of a risk limit violation."""
    timestamp: datetime
    violation_type: str
    severity: str
    current_value: float
    limit_value: float
    message: str


import os

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Daily loss limits - 从环境变量读取，默认 10% daily loss
    daily_loss_limit_pct: float = float(os.getenv("CB_DAILY_LOSS_LIMIT_PCT", "0.10"))
    daily_loss_warning_pct: float = float(os.getenv("CB_DAILY_LOSS_WARNING_PCT", "0.05"))

    # Drawdown limits - 从环境变量读取，默认 15% max drawdown
    max_drawdown_pct: float = float(os.getenv("CB_MAX_DRAWDOWN_PCT", "0.15"))
    drawdown_warning_pct: float = float(os.getenv("CB_DRAWDOWN_WARNING_PCT", "0.08"))

    # VaR limits - 从环境变量读取
    var_95_limit_pct: float = float(os.getenv("CB_VAR_95_LIMIT_PCT", "0.05"))
    var_99_limit_pct: float = float(os.getenv("CB_VAR_99_LIMIT_PCT", "0.10"))

    # Position concentration - 从环境变量读取
    position_concentration_limit: float = float(os.getenv("CB_POSITION_CONCENTRATION_LIMIT", "0.30"))
    concentration_warning_pct: float = float(os.getenv("CB_CONCENTRATION_WARNING_PCT", "0.20"))

    # Cooldown period after state change - 从环境变量读取，默认 5 minutes
    cooldown_period_seconds: int = int(os.getenv("CB_COOLDOWN_SECONDS", "300"))

    # Auto-recovery settings - 从环境变量读取
    auto_recovery_enabled: bool = os.getenv("CB_AUTO_RECOVERY_ENABLED", "true").lower() == "true"
    recovery_check_interval_seconds: int = int(os.getenv("CB_RECOVERY_CHECK_INTERVAL", "60"))

    # Trading limits per state - 从环境变量读取
    normal_position_limit_multiplier: float = float(os.getenv("CB_NORMAL_POS_LIMIT_MULT", "1.0"))
    warning_position_limit_multiplier: float = float(os.getenv("CB_WARNING_POS_LIMIT_MULT", "0.5"))
    restricted_position_limit_multiplier: float = float(os.getenv("CB_RESTRICTED_POS_LIMIT_MULT", "0.1"))

    # VaR model selection: parametric/cornish_fisher/evt/fhs/hybrid
    var_method: str = os.getenv("CB_VAR_METHOD", "hybrid")

    # Per-instrument circuit breaker settings
    enable_per_instrument_limits: bool = os.getenv("CB_ENABLE_PER_INSTRUMENT", "true").lower() == "true"
    per_instrument_notional_limit: float = float(os.getenv("CB_PER_INSTRUMENT_NOTIONAL_LIMIT", "inf"))
    per_instrument_warning_notional: float = float(os.getenv("CB_PER_INSTRUMENT_WARNING_NOTIONAL", "inf"))
    per_instrument_notional_limits: Dict[str, float] = field(default_factory=dict)


def calculate_drawdown(pnl_series: pd.Series) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown from PnL series.

    Args:
        pnl_series: Series of cumulative PnL values

    Returns:
        Tuple of (max_drawdown, drawdown_series)
    """
    if len(pnl_series) == 0:
        return 0.0, pd.Series(dtype=float)

    # Calculate running maximum
    running_max = pnl_series.expanding().max()

    # Calculate drawdown as percentage from running peak.
    # If peak is zero, fall back to unit scale to avoid silent zeros.
    denominator = running_max.abs().where(running_max.abs() > 1e-12, 1.0)
    drawdown = (pnl_series - running_max) / denominator
    drawdown_series = pd.Series(drawdown, index=pnl_series.index)

    return drawdown_series.min(), drawdown_series


@dataclass
class PortfolioState:
    """Current portfolio state for risk checks."""
    timestamp: datetime
    positions: Dict[str, Position]
    cash: float
    pnl_series: pd.Series = field(default_factory=pd.Series)
    asset_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    crypto_balance: float = 0.0
    initial_capital: float = 1.0

    @property
    def positions_notional_value(self) -> float:
        """Gross notional value of all positions using entry prices as proxy marks."""
        return sum(
            abs(position.size) * position.avg_entry_price
            for position in self.positions.values()
        )

    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        # Include position notionals so concentration is measured against portfolio scale.
        return self.cash + self.crypto_balance + self.positions_notional_value

    @property
    def daily_pnl(self) -> float:
        """PnL for current day."""
        if len(self.pnl_series) < 2:
            return 0.0

        # Get today's data
        today = self.timestamp.date()
        today_data = self.pnl_series[
            self.pnl_series.index.date == today
        ]

        if len(today_data) < 2:
            return 0.0

        return today_data.iloc[-1] - today_data.iloc[0]

    @property
    def daily_pnl_pct(self) -> float:
        """Daily PnL as percentage of initial capital."""
        if self.initial_capital == 0:
            return 0.0
        return self.daily_pnl / self.initial_capital

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from PnL series."""
        if self.initial_capital > 0:
            normalized_series = self.pnl_series / self.initial_capital
        else:
            normalized_series = self.pnl_series
        dd, _ = calculate_drawdown(normalized_series)
        return dd

    def get_position_concentration(self) -> Tuple[str, float]:
        """
        Get the largest position concentration.

        Returns:
            Tuple of (instrument, concentration_pct)
        """
        if not self.positions:
            return "", 0.0

        # Concentration risk is a relative cross-position metric.
        # For single-instrument portfolios, this metric is not informative.
        non_zero_positions = [
            pos for pos in self.positions.values()
            if abs(pos.size) > 1e-12 and pos.avg_entry_price > 0
        ]
        if len(non_zero_positions) <= 1:
            return "", 0.0

        total_value = self.total_value
        if abs(total_value) <= 1e-12:
            return "", 0.0
        total_value = abs(total_value)

        max_concentration = 0.0
        max_instrument = ""

        for instrument, position in self.positions.items():
            position_value = abs(position.size) * position.avg_entry_price
            concentration = position_value / total_value

            if concentration > max_concentration:
                max_concentration = concentration
                max_instrument = instrument

        return max_instrument, max_concentration


class CircuitBreaker:
    """
    4-tier circuit breaker system for risk management.

    States:
    - NORMAL: Full trading allowed
    - WARNING: Reduced sizing, tighter monitoring
    - RESTRICTED: Only hedging/liquidation allowed
    - HALTED: All trading suspended

    State transitions:
    - NORMAL → WARNING: Any warning threshold breached
    - WARNING → RESTRICTED: Any limit breached or multiple warnings
    - RESTRICTED → HALTED: Severe breach or multiple limits hit
    - Any state → NORMAL: After cooldown and conditions normalize
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.NORMAL
        # Use deque with maxlen to prevent unlimited growth
        self._max_history_size = 10000
        self.violation_history: deque = deque(maxlen=self._max_history_size)
        self.state_history: deque = deque(maxlen=self._max_history_size)

        # State change tracking
        self._last_state_change: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None

        # Metrics tracking
        self._warning_count: Dict[str, int] = {}
        self._consecutive_warnings: int = 0

        # VaR calculator
        self._var_calculator = VaRCalculator()

        # Thread safety lock
        self._lock = Lock()

        # Redis client for distributed state persistence (optional)
        self._redis_client: Optional[Any] = None
        self._redis_key_prefix = "circuit_breaker"
        self._instrument_states: Dict[str, CircuitState] = {}

    @staticmethod
    def _state_severity(state: CircuitState) -> int:
        """Return numeric severity for safe state comparison."""
        severity = {
            CircuitState.NORMAL: 0,
            CircuitState.WARNING: 1,
            CircuitState.RESTRICTED: 2,
            CircuitState.HALTED: 3,
        }
        return severity[state]

    def set_redis_client(self, redis_client: Any, key_prefix: str = "circuit_breaker") -> None:
        """
        Set Redis client for distributed state persistence.

        Args:
            redis_client: Redis client with get/set methods
            key_prefix: Key prefix for Redis entries
        """
        self._redis_client = redis_client
        self._redis_key_prefix = key_prefix

    async def persist_state(self) -> bool:
        """
        Persist current state to Redis.

        Returns:
            True if successful, False otherwise
        """
        if self._redis_client is None:
            return False

        try:
            import json
            from datetime import timezone

            state_data = {
                "state": self.state.value,
                "last_state_change": self._last_state_change.isoformat() if self._last_state_change else None,
                "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
                "consecutive_warnings": self._consecutive_warnings,
                "warning_count": dict(self._warning_count),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            key = f"{self._redis_key_prefix}:state"
            await self._redis_client.set(
                key,
                json.dumps(state_data),
                ex=3600  # 1 hour expiration
            )
            return True
        except Exception:
            return False

    async def load_state(self) -> bool:
        """
        Load state from Redis.

        Returns:
            True if state was loaded, False otherwise
        """
        if self._redis_client is None:
            return False

        try:
            import json

            key = f"{self._redis_key_prefix}:state"
            data = await self._redis_client.get(key)

            if data is None:
                return False

            state_data = json.loads(data)

            # Restore state
            self.state = CircuitState(state_data["state"])

            if state_data.get("last_state_change"):
                self._last_state_change = datetime.fromisoformat(state_data["last_state_change"])

            if state_data.get("cooldown_until"):
                self._cooldown_until = datetime.fromisoformat(state_data["cooldown_until"])

            self._consecutive_warnings = state_data.get("consecutive_warnings", 0)
            self._warning_count = state_data.get("warning_count", {})

            return True
        except Exception:
            return False

    async def sync_with_redis(self) -> None:
        """
        Synchronize state with Redis.
        Should be called periodically (e.g., every minute) in production.
        """
        # First try to load existing state (for multi-instance consistency)
        loaded = await self.load_state()

        # If no state was loaded, persist our current state
        if not loaded:
            await self.persist_state()

    @property
    def current_state(self) -> CircuitState:
        """Current circuit breaker state."""
        return self.state

    @property
    def is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if self._cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < self._cooldown_until

    def check_risk_limits(self, portfolio: PortfolioState) -> CircuitState:
        """
        Check all risk limits and update state accordingly.

        Thread-safe: uses internal lock to prevent concurrent state modifications.

        Args:
            portfolio: Current portfolio state

        Returns:
            Updated circuit state
        """
        with self._lock:
            violations = self._check_all_limits(portfolio)

            # Record violations
            for violation in violations:
                self._record_violation(violation)

            # Determine new state based on violations
            new_state = self._determine_state(violations)

            # Update state if changed
            if new_state != self.state:
                self._transition_state(new_state, violations)

            return self.state

    def _check_all_limits(self, portfolio: PortfolioState) -> List[Violation]:
        """Check all risk limits and return violations."""
        violations = []
        now = datetime.now(timezone.utc)

        # Check daily loss
        daily_pnl_pct = portfolio.daily_pnl_pct
        if daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            violations.append(Violation(
                timestamp=now,
                violation_type="daily_loss",
                severity="critical",
                current_value=abs(daily_pnl_pct),
                limit_value=self.config.daily_loss_limit_pct,
                message=f"Daily loss {abs(daily_pnl_pct):.2%} exceeds limit {self.config.daily_loss_limit_pct:.2%}"
            ))
        elif daily_pnl_pct <= -self.config.daily_loss_warning_pct:
            violations.append(Violation(
                timestamp=now,
                violation_type="daily_loss",
                severity="warning",
                current_value=abs(daily_pnl_pct),
                limit_value=self.config.daily_loss_warning_pct,
                message=f"Daily loss {abs(daily_pnl_pct):.2%} exceeds warning {self.config.daily_loss_warning_pct:.2%}"
            ))

        # Check drawdown
        max_dd = portfolio.max_drawdown
        if max_dd <= -self.config.max_drawdown_pct:
            violations.append(Violation(
                timestamp=now,
                violation_type="drawdown",
                severity="critical",
                current_value=abs(max_dd),
                limit_value=self.config.max_drawdown_pct,
                message=f"Drawdown {abs(max_dd):.2%} exceeds limit {self.config.max_drawdown_pct:.2%}"
            ))
        elif max_dd <= -self.config.drawdown_warning_pct:
            violations.append(Violation(
                timestamp=now,
                violation_type="drawdown",
                severity="warning",
                current_value=abs(max_dd),
                limit_value=self.config.drawdown_warning_pct,
                message=f"Drawdown {abs(max_dd):.2%} exceeds warning {self.config.drawdown_warning_pct:.2%}"
            ))

        # Check position concentration
        instrument, concentration = portfolio.get_position_concentration()
        if concentration >= self.config.position_concentration_limit:
            violations.append(Violation(
                timestamp=now,
                violation_type="concentration",
                severity="critical",
                current_value=concentration,
                limit_value=self.config.position_concentration_limit,
                message=f"Position {instrument} concentration {concentration:.2%} exceeds limit {self.config.position_concentration_limit:.2%}"
            ))
        elif concentration >= self.config.concentration_warning_pct:
            violations.append(Violation(
                timestamp=now,
                violation_type="concentration",
                severity="warning",
                current_value=concentration,
                limit_value=self.config.concentration_warning_pct,
                message=f"Position {instrument} concentration {concentration:.2%} exceeds warning {self.config.concentration_warning_pct:.2%}"
            ))

        # Check VaR and integrate with breaker.
        var_violation = self._check_var_limit_from_portfolio(portfolio)
        if var_violation is not None:
            violations.append(var_violation)

        # Per-instrument limits
        if self.config.enable_per_instrument_limits:
            violations.extend(self._check_per_instrument_limits(portfolio))

        return violations

    def _check_var_limit_from_portfolio(self, portfolio: PortfolioState) -> Optional[Violation]:
        """Construct VaR inputs from current portfolio snapshot."""
        positions_data: List[Tuple[str, float]] = []
        for instrument, position in portfolio.positions.items():
            if abs(position.size) <= 1e-12 or position.avg_entry_price <= 0:
                continue
            positions_data.append((instrument, abs(position.size) * position.avg_entry_price))

        if not positions_data:
            return None

        positions_df = pd.DataFrame(
            {"value": [v for _, v in positions_data]},
            index=[k for k, _ in positions_data]
        )

        returns_df = portfolio.asset_returns.copy()
        if returns_df.empty:
            # Fallback proxy: portfolio PnL-based return replicated per instrument.
            pnl_returns = portfolio.pnl_series.diff().dropna()
            if len(pnl_returns) < 30:
                return None
            base = max(abs(portfolio.initial_capital), 1e-12)
            norm_returns = pnl_returns / base
            returns_df = pd.DataFrame({
                inst: norm_returns.values for inst, _ in positions_data
            })

        portfolio_value = max(abs(portfolio.total_value), 1e-12)
        return self.check_var_limit(positions_df, returns_df, portfolio_value)

    def _check_per_instrument_limits(self, portfolio: PortfolioState) -> List[Violation]:
        """Instrument-level notional risk checks."""
        now = datetime.now(timezone.utc)
        violations: List[Violation] = []

        for instrument, position in portfolio.positions.items():
            notional = abs(position.size) * max(position.avg_entry_price, 0.0)
            if notional <= 0:
                self._instrument_states[instrument] = CircuitState.NORMAL
                continue

            critical_limit = self.config.per_instrument_notional_limits.get(
                instrument,
                self.config.per_instrument_notional_limit
            )
            warning_limit = min(self.config.per_instrument_warning_notional, critical_limit)

            if not np.isfinite(critical_limit):
                self._instrument_states[instrument] = CircuitState.NORMAL
                continue

            if notional >= critical_limit:
                self._instrument_states[instrument] = CircuitState.RESTRICTED
                violations.append(Violation(
                    timestamp=now,
                    violation_type=f"instrument_notional:{instrument}",
                    severity="critical",
                    current_value=notional,
                    limit_value=critical_limit,
                    message=(
                        f"Instrument {instrument} notional {notional:.2f} "
                        f"exceeds limit {critical_limit:.2f}"
                    )
                ))
            elif np.isfinite(warning_limit) and notional >= warning_limit:
                self._instrument_states[instrument] = CircuitState.WARNING
                violations.append(Violation(
                    timestamp=now,
                    violation_type=f"instrument_notional:{instrument}",
                    severity="warning",
                    current_value=notional,
                    limit_value=warning_limit,
                    message=(
                        f"Instrument {instrument} notional {notional:.2f} "
                        f"exceeds warning {warning_limit:.2f}"
                    )
                ))
            else:
                self._instrument_states[instrument] = CircuitState.NORMAL

        return violations

    def _determine_state(self, violations: List[Violation]) -> CircuitState:
        """Determine appropriate state based on violations."""
        if not violations:
            # No violations - check if we can recover
            if self.state != CircuitState.NORMAL and self._can_recover():
                return self._get_recovery_state()
            return self.state

        # Count severities
        critical_count = sum(1 for v in violations if v.severity == "critical")
        warning_count = sum(1 for v in violations if v.severity == "warning")

        # State determination logic
        if critical_count >= 2 or (critical_count >= 1 and self.state == CircuitState.RESTRICTED):
            return CircuitState.HALTED
        elif critical_count >= 1:
            return CircuitState.RESTRICTED
        elif warning_count >= 2 or (warning_count >= 1 and self.state == CircuitState.WARNING):
            return CircuitState.RESTRICTED
        elif warning_count >= 1:
            return CircuitState.WARNING

        return self.state

    def _can_recover(self) -> bool:
        """Check if we can recover from current state."""
        if self.is_in_cooldown:
            return False

        # Check if enough time has passed since last violation
        if not self.violation_history:
            return True

        last_violation = self.violation_history[-1]
        time_since = datetime.now(timezone.utc) - last_violation.timestamp

        # Require at least cooldown period since last violation
        return time_since.total_seconds() >= self.config.cooldown_period_seconds

    def _get_recovery_state(self) -> CircuitState:
        """Get the state to recover to."""
        # Step down one level at a time
        if self.state == CircuitState.HALTED:
            return CircuitState.RESTRICTED
        elif self.state == CircuitState.RESTRICTED:
            return CircuitState.WARNING
        elif self.state == CircuitState.WARNING:
            return CircuitState.NORMAL
        return CircuitState.NORMAL

    def _transition_state(self, new_state: CircuitState, violations: List[Violation]) -> None:
        """Transition to a new state."""
        now = datetime.now(timezone.utc)

        # Record state change
        reason = "; ".join([v.message for v in violations]) if violations else "Recovery"
        self.state_history.append((now, new_state, reason))

        # Update state
        old_state = self.state
        self.state = new_state
        self._last_state_change = now

        # Set cooldown period
        self._cooldown_until = now + timedelta(seconds=self.config.cooldown_period_seconds)

        # Reset warning counts on state improvement
        if self._state_severity(new_state) < self._state_severity(old_state):
            self._consecutive_warnings = 0
            self._warning_count = {}

        # Send alert on state degradation
        if self._state_severity(new_state) > self._state_severity(old_state):
            self._send_alert(new_state, violations)

        # Redis persistence should be triggered by explicit async maintenance loops.

    def _send_alert(self, state: CircuitState, violations: List[Violation]) -> None:
        """Send alert via logging/webhook when state degrades."""
        import logging
        logger = logging.getLogger(__name__)

        violation_msgs = "; ".join([v.message for v in violations[:3]])
        logger.critical(
            f"CIRCUIT BREAKER ALERT: State changed to {state.value.upper()}. "
            f"Violations: {violation_msgs}"
        )

        # TODO: Add webhook/PagerDuty/Slack integration here
        # webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        # if webhook_url:
        #     asyncio.create_task(self._send_webhook_alert(webhook_url, state, violations))

    async def _send_webhook_alert(self, webhook_url: str, state: CircuitState, violations: List[Violation]) -> None:
        """Send alert to webhook (async)."""
        try:
            import aiohttp
            import json

            payload = {
                "severity": "critical" if state in [CircuitState.RESTRICTED, CircuitState.HALTED] else "warning",
                "state": state.value,
                "violations": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity,
                        "current": v.current_value,
                        "limit": v.limit_value,
                        "message": v.message
                    }
                    for v in violations[:5]
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=5) as response:
                    if response.status >= 400:
                        logger = logging.getLogger(__name__)
                        logger.error(f"Failed to send webhook alert: {response.status}")
        except Exception:
            pass  # Fail silently for alerts

    def _record_violation(self, violation: Violation) -> None:
        """Record a risk limit violation."""
        self.violation_history.append(violation)

        # Track warning counts
        if violation.severity == "warning":
            self._consecutive_warnings += 1
            self._warning_count[violation.violation_type] = \
                self._warning_count.get(violation.violation_type, 0) + 1

    def can_trade(self, action: TradeAction, size: float = 0) -> Tuple[bool, str]:
        """
        Check if a specific trading action is allowed.

        Args:
            action: Type of trading action
            size: Position size (for limit checks)

        Returns:
            Tuple of (allowed, reason)
        """
        if self.state == CircuitState.HALTED:
            return False, "Trading halted - all trading suspended"

        if self.state == CircuitState.RESTRICTED:
            # Only hedging and liquidation allowed
            if action not in [TradeAction.HEDGING, TradeAction.LIQUIDATION]:
                return False, f"Trading restricted - {action.value} not allowed, only hedging/liquidation"

        if self.state == CircuitState.WARNING:
            # Reduced sizing
            if action == TradeAction.NEW_POSITION and size > 0:
                return True, "Warning state - reduced position sizing recommended"

        return True, "Trading allowed"

    def get_position_limit_multiplier(self) -> float:
        """Get position size limit multiplier for current state."""
        multipliers = {
            CircuitState.NORMAL: self.config.normal_position_limit_multiplier,
            CircuitState.WARNING: self.config.warning_position_limit_multiplier,
            CircuitState.RESTRICTED: self.config.restricted_position_limit_multiplier,
            CircuitState.HALTED: 0.0
        }
        return multipliers.get(self.state, 0.0)

    def get_spread_multiplier(self) -> float:
        """Get spread adjustment multiplier for current state."""
        # Widen spreads in higher risk states
        multipliers = {
            CircuitState.NORMAL: 1.0,
            CircuitState.WARNING: 1.5,
            CircuitState.RESTRICTED: 2.0,
            CircuitState.HALTED: float('inf')  # No quotes
        }
        return multipliers.get(self.state, 1.0)

    def manual_reset(self, reason: str = "Manual reset") -> None:
        """
        Manually reset circuit breaker to NORMAL state.

        Args:
            reason: Reason for manual reset
        """
        now = datetime.now(timezone.utc)
        self.state_history.append((now, CircuitState.NORMAL, f"Manual reset: {reason}"))
        self.state = CircuitState.NORMAL
        self._cooldown_until = None
        self._consecutive_warnings = 0
        self._warning_count = {}

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "is_in_cooldown": self.is_in_cooldown,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "last_state_change": self._last_state_change.isoformat() if self._last_state_change else None,
            "violation_count": len(self.violation_history),
            "recent_violations": [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "type": v.violation_type,
                    "severity": v.severity,
                    "message": v.message
                }
                for v in list(self.violation_history)[-5:]  # Last 5
            ],
            "position_limit_multiplier": self.get_position_limit_multiplier(),
            "spread_multiplier": self.get_spread_multiplier(),
            "instrument_states": {k: v.value for k, v in self._instrument_states.items()}
        }

    def check_var_limit(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        portfolio_value: float
    ) -> Optional[Violation]:
        """
        Check VaR against limit.

        Args:
            positions: DataFrame with position values
            returns: DataFrame of historical returns
            portfolio_value: Total portfolio value

        Returns:
            Violation if limit exceeded, None otherwise
        """
        if len(returns) < 30:  # Need sufficient data
            return None

        method = self.config.var_method.lower()
        if method == "parametric":
            var_result = self._var_calculator.parametric_var(positions, returns)
        elif method == "cornish_fisher":
            var_result = self._var_calculator.cornish_fisher_var(positions, returns)
        elif method == "evt":
            var_result = self._var_calculator.evt_var(positions, returns)
        elif method == "fhs":
            var_result = self._var_calculator.filtered_historical_var(positions, returns)
        else:
            # Hybrid: take most conservative 95% VaR across robust estimators.
            candidates = [
                self._var_calculator.parametric_var(positions, returns),
                self._var_calculator.cornish_fisher_var(positions, returns),
                self._var_calculator.evt_var(positions, returns),
                self._var_calculator.filtered_historical_var(positions, returns),
            ]
            var_result = max(candidates, key=lambda x: x.var_95)
        var_pct = var_result.var_95 / portfolio_value if portfolio_value > 0 else 0

        if var_pct > self.config.var_95_limit_pct:
            return Violation(
                timestamp=datetime.now(timezone.utc),
                violation_type="var_95",
                severity="critical" if var_pct > self.config.var_99_limit_pct else "warning",
                current_value=var_pct,
                limit_value=self.config.var_95_limit_pct,
                message=f"VaR 95% {var_pct:.2%} exceeds limit {self.config.var_95_limit_pct:.2%}"
            )

        return None


class CircuitBreakerEnhancedBacktest:
    """
    Enhanced backtest result with circuit breaker metrics.
    """

    def __init__(self, base_result: Any, circuit_breaker: CircuitBreaker):
        self.base_result = base_result
        self.circuit_breaker = circuit_breaker

    @property
    def circuit_breaker_triggers(self) -> int:
        """Number of times circuit breaker was triggered."""
        return len([s for s in self.circuit_breaker.state_history if s[1] != CircuitState.NORMAL])

    @property
    def time_in_each_state(self) -> Dict[str, float]:
        """Time spent in each circuit state (percentage)."""
        # This would need timestamps from backtest
        # Simplified version returns counts
        state_counts = {}
        for _, state, _ in self.circuit_breaker.state_history:
            state_counts[state.value] = state_counts.get(state.value, 0) + 1
        return state_counts

    @property
    def violations_by_type(self) -> Dict[str, int]:
        """Count violations by type."""
        counts = {}
        for v in self.circuit_breaker.violation_history:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts
