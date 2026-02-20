"""
Adaptive Delta Hedger for coin-margined options.

Implements adaptive delta hedging that accounts for:
1. Non-linear PnL characteristics of coin-margined options
2. Price-dependent hedge frequency (accelerate hedging on price drops)
3. Gamma effects on hedge sizing
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, List, Optional, Tuple

import numpy as np

from core.types import Greeks, Position


@dataclass
class AdaptiveHedgeConfig:
    """Configuration for adaptive delta hedger."""
    # Base hedge interval
    base_hedge_interval_minutes: int = 30

    # Price drop thresholds
    price_drop_threshold_pct: float = 0.05  # 5% price drop triggers acceleration
    price_rise_threshold_pct: float = 0.03  # 3% price rise triggers acceleration

    # Hedge frequency multipliers
    hedge_frequency_multiplier: float = 1.5  # Accelerate by 1.5x on price drop
    max_frequency_multiplier: float = 3.0    # Max acceleration

    # Gamma thresholds
    gamma_threshold: float = 0.01  # Gamma threshold for special handling
    high_gamma_multiplier: float = 2.0  # Hedge more frequently when gamma is high

    # Hedge sizing
    min_hedge_size: float = 0.001  # Minimum hedge size
    max_hedge_size_pct: float = 0.5  # Max hedge as % of position

    # Tracking window
    price_history_window: int = 100

    # Inverse option flag
    inverse: bool = True  # Coin-margined by default


@dataclass
class HedgeDecision:
    """Result of hedge decision."""
    should_hedge: bool
    reason: str
    target_delta: float
    current_delta: float
    hedge_size: float
    urgency: str  # "low", "normal", "high", "critical"


class AdaptiveDeltaHedger:
    """
    Adaptive Delta Hedger for coin-margined options.

    Features:
    1. Price-drop acceleration: Hedge 50% more frequently when price drops 5%+
    2. Gamma-adjusted sizing: Increase hedge size when gamma is high
    3. Non-linear PnL awareness: Account for inverse option characteristics
    4. Smart timing: Avoid hedging during temporary price spikes
    """

    def __init__(self, config: AdaptiveHedgeConfig = None):
        self.config = config or AdaptiveHedgeConfig()
        self.last_hedge_time: Optional[datetime] = None
        self.price_history: Deque[Tuple[datetime, float]] = deque(
            maxlen=self.config.price_history_window
        )
        self.hedge_history: List[Tuple[datetime, float, str]] = []
        self._cumulative_price_change: float = 0.0

    def update_price(self, timestamp: datetime, price: float) -> None:
        """Update price history."""
        self.price_history.append((timestamp, price))

    def should_hedge(
        self,
        current_time: datetime,
        current_price: float,
        portfolio_greeks: Greeks,
        position_size: float = 0.0
    ) -> HedgeDecision:
        """
        Determine if hedging is needed and calculate hedge size.

        Args:
            current_time: Current timestamp
            current_price: Current underlying price
            portfolio_greeks: Portfolio Greeks
            position_size: Current position size

        Returns:
            HedgeDecision with details
        """
        # Update price history
        self.update_price(current_time, current_price)

        # Calculate time since last hedge
        time_since_last = self._get_time_since_last_hedge(current_time)

        # Calculate price change metrics
        price_change_pct = self._calculate_price_change_pct(current_price)
        price_drop_pct = self._calculate_price_drop_pct(current_price)
        price_rise_pct = self._calculate_price_rise_pct(current_price)

        # Determine base hedge interval
        base_interval = timedelta(minutes=self.config.base_hedge_interval_minutes)

        # Calculate adjusted interval based on price movement
        adjusted_interval = self._calculate_adjusted_interval(
            base_interval, price_drop_pct, price_rise_pct, portfolio_greeks.gamma
        )

        # Check if we should hedge based on time
        time_trigger = time_since_last >= adjusted_interval

        # Check if we should hedge based on delta deviation
        target_delta = 0.0  # Target is delta-neutral
        current_delta = portfolio_greeks.delta * position_size
        delta_deviation = abs(current_delta - target_delta)

        # Delta threshold increases with time since last hedge
        delta_threshold = self._calculate_delta_threshold(time_since_last, base_interval)
        delta_trigger = delta_deviation > delta_threshold

        # Determine urgency
        urgency = self._determine_urgency(
            price_drop_pct, price_rise_pct, portfolio_greeks.gamma, delta_deviation
        )

        # Make preliminary decision
        should_hedge = time_trigger or delta_trigger or urgency in ["high", "critical"]

        # Calculate hedge size
        if should_hedge:
            hedge_size = self._calculate_hedge_size(
                current_delta, target_delta, current_price, portfolio_greeks.gamma
            )
            # Avoid zero-size "hedges" that only churn state/history.
            if abs(hedge_size) <= 1e-12:
                should_hedge = False
        else:
            hedge_size = 0.0

        # Build reason string
        reason = self._build_reason(
            time_trigger, delta_trigger, urgency, time_since_last, delta_deviation
        )

        return HedgeDecision(
            should_hedge=should_hedge,
            reason=reason,
            target_delta=target_delta,
            current_delta=current_delta,
            hedge_size=hedge_size,
            urgency=urgency
        )

    def _get_time_since_last_hedge(self, current_time: datetime) -> timedelta:
        """Get time elapsed since last hedge."""
        if self.last_hedge_time is None:
            return timedelta(hours=1)  # Large default if never hedged
        return current_time - self.last_hedge_time

    def _calculate_price_change_pct(self, current_price: float) -> float:
        """Calculate price change from reference price."""
        if len(self.price_history) < 2:
            return 0.0

        # Use price from base hedge interval ago as reference
        reference_idx = max(0, len(self.price_history) - 10)
        reference_price = self.price_history[reference_idx][1]

        if reference_price == 0:
            return 0.0

        return (current_price - reference_price) / reference_price

    def _calculate_price_drop_pct(self, current_price: float) -> float:
        """Calculate maximum price drop since last hedge."""
        if len(self.price_history) < 2 or self.last_hedge_time is None:
            return 0.0

        # Find max price since last hedge
        recent_prices = [
            price for ts, price in self.price_history
            if self.last_hedge_time is None or ts > self.last_hedge_time
        ]

        if not recent_prices:
            return 0.0

        max_price = max(recent_prices)
        if max_price == 0:
            return 0.0

        return (max_price - current_price) / max_price

    def _calculate_price_rise_pct(self, current_price: float) -> float:
        """Calculate maximum price rise since last hedge."""
        if len(self.price_history) < 2 or self.last_hedge_time is None:
            return 0.0

        # Find min price since last hedge
        recent_prices = [
            price for ts, price in self.price_history
            if self.last_hedge_time is None or ts > self.last_hedge_time
        ]

        if not recent_prices:
            return 0.0

        min_price = min(recent_prices)
        if min_price == 0:
            return 0.0

        return (current_price - min_price) / min_price

    def _calculate_adjusted_interval(
        self,
        base_interval: timedelta,
        price_drop_pct: float,
        price_rise_pct: float,
        gamma: float
    ) -> timedelta:
        """
        Calculate adjusted hedge interval based on market conditions.

        For coin-margined options:
        - Price drops accelerate hedging (non-linear PnL risk)
        - High gamma accelerates hedging
        """
        multiplier = 1.0

        # Price drop acceleration
        if price_drop_pct >= self.config.price_drop_threshold_pct:
            drop_factor = price_drop_pct / self.config.price_drop_threshold_pct
            multiplier /= (1 + (self.config.hedge_frequency_multiplier - 1) * min(drop_factor, 2))

        # Price rise acceleration (less aggressive)
        if price_rise_pct >= self.config.price_rise_threshold_pct:
            rise_factor = price_rise_pct / self.config.price_rise_threshold_pct
            multiplier /= (1 + 0.5 * min(rise_factor, 2))  # 0.5x less aggressive than drops

        # Gamma acceleration
        if abs(gamma) > self.config.gamma_threshold:
            gamma_factor = min(abs(gamma) / self.config.gamma_threshold, 3)
            multiplier /= (1 + (self.config.high_gamma_multiplier - 1) * (gamma_factor - 1) / 2)

        # Clamp multiplier
        multiplier = max(1.0 / self.config.max_frequency_multiplier, min(multiplier, 2.0))

        adjusted_seconds = base_interval.total_seconds() * multiplier
        return timedelta(seconds=adjusted_seconds)

    def _calculate_delta_threshold(
        self,
        time_since_last: timedelta,
        base_interval: timedelta
    ) -> float:
        """
        Calculate delta deviation threshold.

        Threshold decreases as time since last hedge increases.
        """
        base_threshold = 0.05  # 5% delta deviation

        # Reduce threshold as time passes
        time_factor = time_since_last / base_interval
        threshold = base_threshold / max(1, time_factor ** 0.5)

        return threshold

    def _determine_urgency(
        self,
        price_drop_pct: float,
        price_rise_pct: float,
        gamma: float,
        delta_deviation: float
    ) -> str:
        """Determine urgency level for hedging."""
        # Critical: Large price drop with high gamma
        if price_drop_pct >= 0.10 or (price_drop_pct >= 0.05 and abs(gamma) > 0.02):
            return "critical"

        # High: Significant price drop or large delta deviation
        if price_drop_pct >= 0.05 or delta_deviation > 0.15:
            return "high"

        # Normal: Moderate conditions
        if price_drop_pct >= 0.03 or price_rise_pct >= 0.05 or delta_deviation > 0.08:
            return "normal"

        return "low"

    def _calculate_hedge_size(
        self,
        current_delta: float,
        target_delta: float,
        current_price: float,
        gamma: float
    ) -> float:
        """
        Calculate hedge size considering coin-margined non-linearity.

        For coin-margined options:
        - Hedge size needs to account for inverse price relationship
        - Gamma effects are more pronounced at lower prices
        """
        delta_diff = target_delta - current_delta

        # Base hedge size
        base_size = abs(delta_diff)

        # Adjust for coin-margined non-linearity
        if self.config.inverse and current_price > 0:
            # At lower prices, same delta change requires larger size adjustment
            # due to 1/S relationship
            inverse_factor = 50000 / current_price  # Normalize to $50k BTC
            base_size *= min(inverse_factor, 2.0)  # Cap at 2x

        # Adjust for gamma
        if abs(gamma) > self.config.gamma_threshold:
            # High gamma: hedge more conservatively (smaller steps)
            gamma_adjustment = 1 / (1 + abs(gamma) * 10)
            base_size *= gamma_adjustment

        # Apply limits
        hedge_size = max(self.config.min_hedge_size, base_size)
        max_size = abs(current_delta) * self.config.max_hedge_size_pct
        hedge_size = min(hedge_size, max_size)

        # Preserve sign
        return np.sign(delta_diff) * hedge_size

    def _build_reason(
        self,
        time_trigger: bool,
        delta_trigger: bool,
        urgency: str,
        time_since_last: timedelta,
        delta_deviation: float
    ) -> str:
        """Build human-readable reason for hedge decision."""
        reasons = []

        if urgency in ["critical", "high"]:
            reasons.append(f"Urgency: {urgency}")

        if time_trigger:
            reasons.append(f"Time trigger ({time_since_last.total_seconds()/60:.1f}min)")

        if delta_trigger:
            reasons.append(f"Delta deviation ({delta_deviation:.4f})")

        if not reasons:
            return "No hedge needed"

        return "; ".join(reasons)

    def execute_hedge(self, timestamp: datetime, hedge_size: float, price: float) -> None:
        """
        Record hedge execution.

        Args:
            timestamp: Execution time
            hedge_size: Size of hedge executed
            price: Price at execution
        """
        self.last_hedge_time = timestamp
        self.hedge_history.append((timestamp, hedge_size, f"@{price:.2f}"))

    def get_hedge_stats(self) -> dict:
        """Get hedging statistics."""
        if not self.hedge_history:
            return {
                "total_hedges": 0,
                "avg_hedge_size": 0.0,
                "last_hedge_time": None
            }

        hedge_sizes = [abs(h[1]) for h in self.hedge_history]

        return {
            "total_hedges": len(self.hedge_history),
            "avg_hedge_size": np.mean(hedge_sizes),
            "max_hedge_size": max(hedge_sizes),
            "min_hedge_size": min(hedge_sizes),
            "last_hedge_time": self.last_hedge_time.isoformat() if self.last_hedge_time else None,
            "time_since_last_hedge_minutes": (
                (datetime.now(timezone.utc) - self.last_hedge_time).total_seconds() / 60
                if self.last_hedge_time else None
            )
        }

    def reset(self) -> None:
        """Reset hedger state."""
        self.last_hedge_time = None
        self.price_history.clear()
        self.hedge_history.clear()
        self._cumulative_price_change = 0.0


class SimpleDeltaHedger:
    """
    Simple periodic delta hedger (baseline for comparison).
    """

    def __init__(self, hedge_interval_minutes: int = 30):
        self.hedge_interval = timedelta(minutes=hedge_interval_minutes)
        self.last_hedge_time: Optional[datetime] = None

    def should_hedge(
        self,
        current_time: datetime,
        current_price: float,
        portfolio_greeks: Greeks,
        position_size: float = 0.0
    ) -> HedgeDecision:
        """Simple time-based hedge decision."""
        if self.last_hedge_time is None:
            should_hedge = True
            reason = "First hedge"
        else:
            time_since = current_time - self.last_hedge_time
            should_hedge = time_since >= self.hedge_interval
            reason = f"Time-based ({time_since.total_seconds()/60:.1f}min)" if should_hedge else "No hedge needed"

        target_delta = 0.0
        current_delta = portfolio_greeks.delta * position_size
        hedge_size = target_delta - current_delta if should_hedge else 0.0

        return HedgeDecision(
            should_hedge=should_hedge,
            reason=reason,
            target_delta=target_delta,
            current_delta=current_delta,
            hedge_size=hedge_size,
            urgency="normal" if should_hedge else "low"
        )

    def execute_hedge(self, timestamp: datetime, hedge_size: float, price: float) -> None:
        """Record hedge execution."""
        self.last_hedge_time = timestamp

