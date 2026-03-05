"""Circuit breaker system for portfolio risk management."""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
import os
from threading import Lock, Thread
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.types import Position, Portfolio
from research.risk.var import VaRCalculator, VaRResult

logger = logging.getLogger(__name__)

REDIS_STATE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    RuntimeError,
    OSError,
    json.JSONDecodeError,
)
ALERT_RUNNER_EXCEPTIONS = (RuntimeError, TypeError, ValueError, OSError)


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


def _state_severity(state: CircuitState) -> int:
    """Return numeric severity for safe state comparison."""
    severity = {
        CircuitState.NORMAL: 0,
        CircuitState.WARNING: 1,
        CircuitState.RESTRICTED: 2,
        CircuitState.HALTED: 3,
    }
    return severity[state]


def _alert_severity(state: CircuitState) -> str:
    """Map circuit state to alert severity label."""
    return "critical" if state in [CircuitState.RESTRICTED, CircuitState.HALTED] else "warning"


def _build_threshold_violation(
    *,
    now: datetime,
    violation_type: str,
    value: float,
    critical_limit: float,
    warning_limit: float,
    label: str,
) -> Optional["Violation"]:
    if value >= critical_limit:
        return Violation(
            timestamp=now,
            violation_type=violation_type,
            severity="critical",
            current_value=value,
            limit_value=critical_limit,
            message=f"{label} {value:.2%} exceeds limit {critical_limit:.2%}",
        )
    if value >= warning_limit:
        return Violation(
            timestamp=now,
            violation_type=violation_type,
            severity="warning",
            current_value=value,
            limit_value=warning_limit,
            message=f"{label} {value:.2%} exceeds warning {warning_limit:.2%}",
        )
    return None


def _daily_loss_violation_for_portfolio(
    *, portfolio: "PortfolioState", config: "CircuitBreakerConfig", now: datetime
) -> Optional["Violation"]:
    """Build daily-loss violation if daily PnL is negative and breaches thresholds."""
    daily_pnl_pct = portfolio.daily_pnl_pct
    if daily_pnl_pct >= 0:
        return None
    return _build_threshold_violation(
        now=now,
        violation_type="daily_loss",
        value=abs(daily_pnl_pct),
        critical_limit=config.daily_loss_limit_pct,
        warning_limit=config.daily_loss_warning_pct,
        label="Daily loss",
    )


def _drawdown_violation_for_portfolio(
    *, portfolio: "PortfolioState", config: "CircuitBreakerConfig", now: datetime
) -> Optional["Violation"]:
    """Build drawdown violation if max drawdown is negative and breaches thresholds."""
    max_dd = portfolio.max_drawdown
    if max_dd >= 0:
        return None
    return _build_threshold_violation(
        now=now,
        violation_type="drawdown",
        value=abs(max_dd),
        critical_limit=config.max_drawdown_pct,
        warning_limit=config.drawdown_warning_pct,
        label="Drawdown",
    )


def _concentration_violation_for_portfolio(
    *, portfolio: "PortfolioState", config: "CircuitBreakerConfig", now: datetime
) -> Optional["Violation"]:
    """Build concentration violation for largest instrument exposure."""
    instrument, concentration = portfolio.get_position_concentration()
    return _build_threshold_violation(
        now=now,
        violation_type="concentration",
        value=concentration,
        critical_limit=config.position_concentration_limit,
        warning_limit=config.concentration_warning_pct,
        label=f"Position {instrument} concentration",
    )


def _should_halt_from_critical(critical_count: int, state: "CircuitState") -> bool:
    """Whether current critical violations imply an immediate HALTED state."""
    return critical_count >= 2 or (critical_count >= 1 and state == CircuitState.RESTRICTED)


def _should_restrict_from_warning(warning_count: int, state: "CircuitState") -> bool:
    """Whether warning violations imply escalation to RESTRICTED."""
    return warning_count >= 2 or (warning_count >= 1 and state == CircuitState.WARNING)


def _portfolio_signed_notionals(portfolio: "PortfolioState") -> List[Tuple[str, float]]:
    """Extract per-instrument signed notionals from a live portfolio snapshot."""
    positions_data: List[Tuple[str, float]] = []
    for instrument, position in portfolio.positions.items():
        if abs(position.size) <= 1e-12 or position.avg_entry_price <= 0:
            continue
        signed_notional = float(position.size) * float(position.avg_entry_price)
        positions_data.append((instrument, signed_notional))
    return positions_data


def _build_proxy_var_inputs(
    portfolio: "PortfolioState",
    positions_data: List[Tuple[str, float]],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build single-factor proxy inputs when no per-asset return matrix exists."""
    pnl_returns = portfolio.pnl_series.diff().dropna()
    if len(pnl_returns) < 30:
        return None

    base = max(abs(portfolio.initial_capital), 1e-12)
    norm_returns = pnl_returns / base
    proxy_value = float(sum(value for _, value in positions_data))
    if abs(proxy_value) <= 1e-12:
        return None

    positions_df = pd.DataFrame({"value": [proxy_value]}, index=["_portfolio_proxy"])
    returns_df = pd.DataFrame({"_portfolio_proxy": norm_returns.values})
    return positions_df, returns_df


def _build_instrument_var_inputs(
    positions_data: List[Tuple[str, float]],
    returns_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    positions_df = pd.DataFrame(
        {"value": [value for _, value in positions_data]},
        index=[instrument for instrument, _ in positions_data],
    )
    return positions_df, returns_df


def _evaluate_instrument_notional_limit(
    *,
    instrument: str,
    position: "Position",
    config: "CircuitBreakerConfig",
    now: datetime,
) -> Tuple["CircuitState", Optional["Violation"]]:
    notional = abs(position.size) * max(position.avg_entry_price, 0.0)
    if notional <= 0:
        return CircuitState.NORMAL, None
    critical_limit = config.per_instrument_notional_limits.get(instrument, config.per_instrument_notional_limit)
    if not np.isfinite(critical_limit):
        return CircuitState.NORMAL, None
    warning_limit = min(config.per_instrument_warning_notional, critical_limit)
    for severity, limit, state, label in (
        ("critical", critical_limit, CircuitState.RESTRICTED, "limit"),
        ("warning", warning_limit, CircuitState.WARNING, "warning"),
    ):
        if not np.isfinite(limit) or notional < limit:
            continue
        return state, Violation(timestamp=now, violation_type=f"instrument_notional:{instrument}", severity=severity, current_value=notional, limit_value=limit, message=f"Instrument {instrument} notional {notional:.2f} exceeds {label} {limit:.2f}")
    return CircuitState.NORMAL, None


def _select_var_result(
    *,
    calculator: VaRCalculator,
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    method: str,
) -> VaRResult:
    method_key = method.lower()
    method_map = {
        "parametric": calculator.parametric_var,
        "cornish_fisher": calculator.cornish_fisher_var,
        "evt": calculator.evt_var,
        "fhs": calculator.filtered_historical_var,
    }
    selected = method_map.get(method_key)
    if selected is not None:
        return selected(positions, returns)
    candidates = [
        calculator.parametric_var(positions, returns),
        calculator.cornish_fisher_var(positions, returns),
        calculator.evt_var(positions, returns),
        calculator.filtered_historical_var(positions, returns),
    ]
    return max(candidates, key=lambda result: result.var_95)


def _var_percentages(var_result: VaRResult, portfolio_value: float) -> Tuple[float, float]:
    if portfolio_value <= 0:
        return 0.0, 0.0
    return var_result.var_95 / portfolio_value, var_result.var_99 / portfolio_value


def _var_limit_violation(
    *,
    config: "CircuitBreakerConfig",
    var_95_pct: float,
    var_99_pct: float,
    now: datetime,
) -> Optional["Violation"]:
    if var_99_pct > config.var_99_limit_pct:
        return Violation(
            timestamp=now,
            violation_type="var_99",
            severity="critical",
            current_value=var_99_pct,
            limit_value=config.var_99_limit_pct,
            message=f"VaR 99% {var_99_pct:.2%} exceeds limit {config.var_99_limit_pct:.2%}",
        )
    if var_95_pct > config.var_95_limit_pct:
        return Violation(
            timestamp=now,
            violation_type="var_95",
            severity="warning",
            current_value=var_95_pct,
            limit_value=config.var_95_limit_pct,
            message=f"VaR 95% {var_95_pct:.2%} exceeds limit {config.var_95_limit_pct:.2%}",
        )
    return None


def _schedule_webhook_alert_method(
    self: Any,
    webhook_url: str,
    state: CircuitState,
    violations: List["Violation"],
) -> None:
    """Schedule generic webhook alert."""
    self._schedule_async_alert(
        lambda: self._send_webhook_alert(webhook_url, state, violations)
    )


def _schedule_slack_alert_method(
    self: Any,
    webhook_url: str,
    state: CircuitState,
    violations: List["Violation"],
) -> None:
    """Schedule Slack-compatible webhook alert."""
    self._schedule_async_alert(
        lambda: self._send_slack_alert(webhook_url, state, violations)
    )


@dataclass
class Violation:
    """Record of a risk limit violation."""
    timestamp: datetime
    violation_type: str
    severity: str
    current_value: float
    limit_value: float
    message: str

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

    # Alert integrations
    alert_enabled: bool = os.getenv("CB_ALERT_ENABLED", "true").lower() == "true"
    alert_webhook_url: str = os.getenv("CB_ALERT_WEBHOOK_URL", os.getenv("ALERT_WEBHOOK_URL", ""))
    slack_webhook_url: str = os.getenv("CB_SLACK_WEBHOOK_URL", "")
    alert_timeout_seconds: float = float(os.getenv("CB_ALERT_TIMEOUT_SECONDS", "5"))


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
        """PnL over trailing 24h window ending at current timestamp."""
        if len(self.pnl_series) < 2:
            return 0.0

        if not isinstance(self.pnl_series.index, pd.DatetimeIndex):
            return 0.0

        window_end = pd.Timestamp(self.timestamp)
        index_tz = self.pnl_series.index.tz
        if index_tz is not None and window_end.tzinfo is None:
            window_end = window_end.tz_localize(index_tz)
        elif index_tz is None and window_end.tzinfo is not None:
            window_end = window_end.tz_localize(None)
        window_start = window_end - pd.Timedelta(hours=24)

        window_mask = (self.pnl_series.index >= window_start) & (self.pnl_series.index <= window_end)
        today_data = self.pnl_series[window_mask]

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
        if not self.positions: return "", 0.0
        non_zero_positions = [pos for pos in self.positions.values() if abs(pos.size) > 1e-12 and pos.avg_entry_price > 0]
        if len(non_zero_positions) <= 1: return "", 0.0
        total_value = self.total_value
        if abs(total_value) <= 1e-12: return "", 0.0
        total_value = abs(total_value)
        max_concentration, max_instrument = 0.0, ""
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
        except REDIS_STATE_EXCEPTIONS:
            logger.exception("Failed to persist circuit breaker state to Redis")
            return False

    async def load_state(self) -> bool:
        """Load state snapshot from Redis if available."""
        if self._redis_client is None:
            return False

        try:
            key = f"{self._redis_key_prefix}:state"; data = await self._redis_client.get(key)

            if data is None:
                return False

            state_data = json.loads(data)
            self.state = CircuitState(state_data["state"])

            if state_data.get("last_state_change"):
                self._last_state_change = datetime.fromisoformat(state_data["last_state_change"])

            if state_data.get("cooldown_until"):
                self._cooldown_until = datetime.fromisoformat(state_data["cooldown_until"])

            self._consecutive_warnings = state_data.get("consecutive_warnings", 0)
            self._warning_count = state_data.get("warning_count", {})

            return True
        except REDIS_STATE_EXCEPTIONS:
            logger.exception("Failed to load circuit breaker state from Redis")
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
        violations = []; now = datetime.now(timezone.utc)
        for violation in (
            _daily_loss_violation_for_portfolio(portfolio=portfolio, config=self.config, now=now),
            _drawdown_violation_for_portfolio(portfolio=portfolio, config=self.config, now=now),
            _concentration_violation_for_portfolio(portfolio=portfolio, config=self.config, now=now),
        ):
            if violation is not None:
                violations.append(violation)
        if (var_violation := self._check_var_limit_from_portfolio(portfolio)) is not None: violations.append(var_violation)
        if self.config.enable_per_instrument_limits: violations.extend(self._check_per_instrument_limits(portfolio))
        return violations

    def _check_var_limit_from_portfolio(self, portfolio: PortfolioState) -> Optional[Violation]:
        """Construct VaR inputs from current portfolio snapshot."""
        positions_data = _portfolio_signed_notionals(portfolio)
        if not positions_data:
            return None

        returns_df = portfolio.asset_returns.copy()
        if returns_df.empty:
            proxy_inputs = _build_proxy_var_inputs(portfolio, positions_data)
            if proxy_inputs is None:
                return None
            positions_df, returns_df = proxy_inputs
        else:
            positions_df, returns_df = _build_instrument_var_inputs(positions_data, returns_df)

        portfolio_value = max(abs(portfolio.total_value), 1e-12)
        return self.check_var_limit(positions_df, returns_df, portfolio_value)

    def _check_per_instrument_limits(self, portfolio: PortfolioState) -> List[Violation]:
        now = datetime.now(timezone.utc)
        violations: List[Violation] = []
        for instrument, position in portfolio.positions.items():
            state, violation = _evaluate_instrument_notional_limit(
                instrument=instrument,
                position=position,
                config=self.config,
                now=now,
            )
            self._instrument_states[instrument] = state
            if violation is not None:
                violations.append(violation)
        return violations

    def _determine_state(self, violations: List[Violation]) -> CircuitState:
        """Determine appropriate state based on violations."""
        if not violations:
            if self.state != CircuitState.NORMAL and self._can_recover():
                return self._get_recovery_state()
            return self.state

        critical_count = sum(1 for v in violations if v.severity == "critical")
        warning_count = sum(1 for v in violations if v.severity == "warning")
        if _should_halt_from_critical(critical_count, self.state):
            return CircuitState.HALTED
        if critical_count >= 1:
            return CircuitState.RESTRICTED
        if _should_restrict_from_warning(warning_count, self.state):
            return CircuitState.RESTRICTED
        if warning_count >= 1:
            return CircuitState.WARNING
        return self.state

    def _can_recover(self) -> bool:
        """Check if we can recover from current state."""
        if self.is_in_cooldown:
            return False

        if not self.violation_history:
            return True

        last_violation = self.violation_history[-1]
        time_since = datetime.now(timezone.utc) - last_violation.timestamp

        return time_since.total_seconds() >= self.config.cooldown_period_seconds

    def _get_recovery_state(self) -> CircuitState:
        """Get the state to recover to."""
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
        reason = "; ".join([v.message for v in violations]) if violations else "Recovery"
        self.state_history.append((now, new_state, reason))
        old_state = self.state
        self.state = new_state
        self._last_state_change = now
        self._cooldown_until = now + timedelta(seconds=self.config.cooldown_period_seconds)
        if _state_severity(new_state) < _state_severity(old_state):
            self._consecutive_warnings = 0
            self._warning_count = {}
        if _state_severity(new_state) > _state_severity(old_state):
            self._send_alert(new_state, violations)

        # Redis persistence should be triggered by explicit async maintenance loops.

    def _send_alert(self, state: CircuitState, violations: List[Violation]) -> None:
        """Send alert via logging/webhook when state degrades."""
        violation_msgs = "; ".join([v.message for v in violations[:3]])
        logger.critical(
            f"CIRCUIT BREAKER ALERT: State changed to {state.value.upper()}. "
            f"Violations: {violation_msgs}"
        )

        if not self.config.alert_enabled:
            return

        webhook_url = (self.config.alert_webhook_url or "").strip()
        if webhook_url:
            self._schedule_webhook_alert(webhook_url, state, violations)

        slack_webhook_url = (self.config.slack_webhook_url or "").strip()
        if slack_webhook_url:
            self._schedule_slack_alert(slack_webhook_url, state, violations)

    def _schedule_async_alert(
        self,
        coroutine_factory: Callable[[], Awaitable[None]],
    ) -> None:
        """Schedule async alert delivery without blocking risk checks."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coroutine_factory())
            return
        except RuntimeError:
            loop = None
        if loop is not None:
            return

        def _runner() -> None:
            try:
                asyncio.run(coroutine_factory())
            except ALERT_RUNNER_EXCEPTIONS:
                logger.exception("Background alert delivery failed")

        Thread(target=_runner, daemon=True).start()

    def _build_alert_payload(self, state: CircuitState, violations: List[Violation]) -> Dict[str, Any]:
        """Build structured payload for generic webhook integrations."""
        return {
            "severity": _alert_severity(state),
            "state": state.value,
            "violation_count": len(violations),
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "current": v.current_value,
                    "limit": v.limit_value,
                    "message": v.message,
                }
                for v in violations[:5]
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_slack_payload(self, state: CircuitState, violations: List[Violation]) -> Dict[str, Any]:
        """Build Slack webhook payload."""
        top_msgs = [f"- {v.message}" for v in violations[:5]]
        lines = [
            f"*Circuit Breaker Alert*: `{state.value.upper()}`",
            f"*Severity*: `{_alert_severity(state).upper()}`",
            f"*Violation Count*: {len(violations)}",
            "*Top Violations:*",
        ]
        if top_msgs:
            lines.extend(top_msgs)
        else:
            lines.append("- (none)")

        return {
            "text": "\n".join(lines),
            "username": "Risk Circuit Breaker",
            "icon_emoji": ":rotating_light:",
        }

    async def _send_webhook_alert(
        self,
        webhook_url: str,
        state: CircuitState,
        violations: List[Violation],
    ) -> None:
        """Send alert to webhook (async)."""
        try:
            import aiohttp
        except ImportError:
            logger.exception("Failed to deliver webhook alert")
            return
        client_error = getattr(aiohttp, "ClientError", RuntimeError)

        try:
            payload = self._build_alert_payload(state, violations)
            timeout = aiohttp.ClientTimeout(total=max(self.config.alert_timeout_seconds, 0.1))

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=timeout) as response:
                    if response.status >= 400:
                        logger.error("Failed to send webhook alert: %s", response.status)
        except (client_error, asyncio.TimeoutError, RuntimeError, ValueError, TypeError, OSError):
            logger.exception("Failed to deliver webhook alert")

    async def _send_slack_alert(
        self,
        webhook_url: str,
        state: CircuitState,
        violations: List[Violation],
    ) -> None:
        """Send alert to Slack incoming webhook (async)."""
        try:
            import aiohttp
        except ImportError:
            logger.exception("Failed to deliver Slack alert")
            return
        client_error = getattr(aiohttp, "ClientError", RuntimeError)

        try:
            payload = self._build_slack_payload(state, violations)
            timeout = aiohttp.ClientTimeout(total=max(self.config.alert_timeout_seconds, 0.1))

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=timeout) as response:
                    if response.status >= 400:
                        logger.error("Failed to send Slack alert: %s", response.status)
        except (client_error, asyncio.TimeoutError, RuntimeError, ValueError, TypeError, OSError):
            logger.exception("Failed to deliver Slack alert")

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

    def check_var_limit(self, positions: pd.DataFrame, returns: pd.DataFrame, portfolio_value: float) -> Optional[Violation]:
        """Check configured VaR thresholds and return a violation when breached."""
        if len(returns) < 30:
            return None
        var_result = _select_var_result(
            calculator=self._var_calculator,
            positions=positions,
            returns=returns,
            method=self.config.var_method,
        )
        var_95_pct, var_99_pct = _var_percentages(var_result, portfolio_value)
        return _var_limit_violation(
            config=self.config,
            var_95_pct=var_95_pct,
            var_99_pct=var_99_pct,
            now=datetime.now(timezone.utc),
        )


CircuitBreaker._schedule_webhook_alert = _schedule_webhook_alert_method
CircuitBreaker._schedule_slack_alert = _schedule_slack_alert_method


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
