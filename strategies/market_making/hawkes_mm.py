"""
Hawkes process-based market making strategy.

This strategy uses Hawkes process to model trade arrival clustering and dynamically
adjusts quotes based on predicted market activity.

Key features:
1. Real-time Hawkes intensity estimation
2. Dynamic spread adjustment based on predicted activity
3. Inventory skew with activity-aware risk management
4. Adverse selection detection through intensity anomalies

References:
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). Algorithmic and high-frequency trading.
- Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). Hawkes processes in finance.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

from core.types import MarketState, Position, QuoteAction
from strategies.base import MarketMakingStrategy
from data.generators.hawkes import HawkesParameters


@dataclass
class HawkesMMConfig:
    """Configuration for Hawkes-based market maker.

    All parameters can be set via environment variables.
    """
    # Base spread parameters
    base_spread_bps: float = field(default_factory=lambda: float(__import__('os').getenv("MM_BASE_SPREAD_BPS", "20.0")))
    min_spread_bps: float = field(default_factory=lambda: float(__import__('os').getenv("MM_MIN_SPREAD_BPS", "5.0")))
    max_spread_bps: float = field(default_factory=lambda: float(__import__('os').getenv("MM_MAX_SPREAD_BPS", "100.0")))

    # Quote size
    quote_size: float = field(default_factory=lambda: float(__import__('os').getenv("MM_QUOTE_SIZE", "1.0")))
    inventory_limit: float = field(default_factory=lambda: float(__import__('os').getenv("MM_INVENTORY_LIMIT", "10.0")))

    # Hawkes parameters (estimated from historical data or set manually)
    hawkes_mu: float = field(default_factory=lambda: float(__import__('os').getenv("HAWKES_MU", "0.1")))
    hawkes_alpha: float = field(default_factory=lambda: float(__import__('os').getenv("HAWKES_ALPHA", "0.4")))
    hawkes_beta: float = field(default_factory=lambda: float(__import__('os').getenv("HAWKES_BETA", "0.8")))

    # Intensity-based adjustment
    high_intensity_threshold: float = field(default_factory=lambda: float(__import__('os').getenv("MM_HIGH_INTENSITY_THRESH", "0.3")))
    low_intensity_threshold: float = field(default_factory=lambda: float(__import__('os').getenv("MM_LOW_INTENSITY_THRESH", "0.05")))

    # Spread adjustment factors
    aggressive_factor: float = field(default_factory=lambda: float(__import__('os').getenv("MM_AGGRESSIVE_FACTOR", "0.5")))  # Narrow spread when active
    defensive_factor: float = field(default_factory=lambda: float(__import__('os').getenv("MM_DEFENSIVE_FACTOR", "2.0")))   # Widen spread when quiet

    # Inventory management
    inventory_skew_factor: float = field(default_factory=lambda: float(__import__('os').getenv("MM_INVENTORY_SKEW", "0.5")))

    # Online estimation window (number of recent trades to use for estimation)
    estimation_window: int = field(default_factory=lambda: int(__import__('os').getenv("HAWKES_EST_WINDOW", "100")))

    # Adverse selection detection
    adverse_selection_threshold: float = field(default_factory=lambda: float(__import__('os').getenv("MM_ADV_SEL_THRESH", "2.0")))
    mark_power: float = field(default_factory=lambda: float(__import__('os').getenv("HAWKES_MARK_POWER", "0.5")))
    enable_online_mle: bool = field(default_factory=lambda: __import__('os').getenv("HAWKES_ENABLE_ONLINE_MLE", "1") != "0")


class HawkesIntensityMonitor:
    """Monitor and estimate Hawkes intensity from recent trade flow.

    This class maintains a sliding window of recent trades and computes
    the conditional intensity λ(t) in real-time.
    """

    def __init__(self, params: HawkesParameters, window_size: int = 100, mark_power: float = 0.5):
        """
        Initialize intensity monitor.

        Args:
            params: Hawkes process parameters
            window_size: Number of recent trades to maintain
        """
        self.params = params
        self.window_size = window_size
        self.mark_power = max(0.1, float(mark_power))
        self.trade_times: deque = deque(maxlen=window_size)
        self.trade_directions: deque = deque(maxlen=window_size)
        self.trade_sizes: deque = deque(maxlen=window_size)
        self._A: float = 0.0  # recursive kernel accumulator (total)
        self._A_buy: float = 0.0
        self._A_sell: float = 0.0
        self._last_event_time: float = 0.0

    def add_trade(self, timestamp: float, direction: int = 0, size: float = 1.0):
        """Record a new trade and update O(1) recursive intensity.

        Args:
            timestamp: Trade timestamp
            direction: +1 for buyer-initiated, -1 for seller-initiated, 0 for unknown
            size: Trade size (mark) used for marked excitation
        """
        if self._last_event_time > 0:
            dt = max(0.0, timestamp - self._last_event_time)
            decay = np.exp(-self.params.beta * dt)
            self._A *= decay
            self._A_buy *= decay
            self._A_sell *= decay

        mark = max(float(size), 1e-8) ** self.mark_power
        excitation = self.params.alpha * mark

        # Apply current event excitation
        self._A += excitation
        if direction > 0:
            self._A_buy += excitation
        elif direction < 0:
            self._A_sell += excitation
        else:
            self._A_buy += excitation * 0.5
            self._A_sell += excitation * 0.5

        self._last_event_time = timestamp
        self.trade_times.append(timestamp)
        self.trade_directions.append(direction)
        self.trade_sizes.append(float(size))

    def get_intensity(self, current_time: float) -> float:
        """Compute conditional intensity λ(t) at current time in O(1).

        Uses recursive formula: A(t) = exp(-β(t - t_last)) * A_last
        λ(t) = μ + A(t)

        Args:
            current_time: Current timestamp

        Returns:
            Conditional intensity
        """
        if self._last_event_time <= 0 or not self.trade_times:
            return self.params.mu

        dt = current_time - self._last_event_time
        A_now = np.exp(-self.params.beta * max(0.0, dt)) * self._A
        return self.params.mu + A_now

    def get_buy_sell_intensity(self, current_time: float) -> Tuple[float, float]:
        """Compute separate intensities for buy and sell trades.

        This helps detect order flow imbalance.

        Returns:
            (buy_intensity, sell_intensity)
        """
        if self._last_event_time <= 0 or not self.trade_times:
            return self.params.mu / 2, self.params.mu / 2

        dt = max(0.0, current_time - self._last_event_time)
        decay = np.exp(-self.params.beta * dt)
        buy_intensity = self.params.mu / 2 + decay * self._A_buy
        sell_intensity = self.params.mu / 2 + decay * self._A_sell
        return float(buy_intensity), float(sell_intensity)

    def detect_adverse_selection(self, current_time: float, price_change: float) -> bool:
        """Detect potential adverse selection.

        Adverse selection occurs when:
        1. High trade intensity AND
        2. Price moves against our position

        Args:
            current_time: Current timestamp
            price_change: Recent price change (positive = up, negative = down)

        Returns:
            True if adverse selection detected
        """
        if len(self.trade_times) < 10:
            return False

        intensity = self.get_intensity(current_time)
        buy_int, sell_int = self.get_buy_sell_intensity(current_time)

        # Check for intensity spike
        expected_intensity = self.params.long_term_intensity
        if intensity < expected_intensity * 2:  # Not intense enough
            return False

        # Check for directional imbalance
        if buy_int > sell_int * 2 and price_change < 0:
            # Many buys but price dropping = potential adverse selection for sellers
            return True
        elif sell_int > buy_int * 2 and price_change > 0:
            # Many sells but price rising = potential adverse selection for buyers
            return True

        return False

    def _estimate_parameters_moments(self) -> Optional[HawkesParameters]:
        """Method-of-moments fallback estimator."""
        if len(self.trade_times) < 20:
            return None

        times = np.array(self.trade_times, dtype=float)
        iet = np.diff(times)
        if len(iet) < 2:
            return None

        mean_iet = np.mean(iet)
        var_iet = np.var(iet)
        cv = np.sqrt(var_iet) / mean_iet if mean_iet > 0 else 1.0

        if cv > 1.2:
            branching_ratio = min(0.8, (cv - 1) / cv)
        else:
            branching_ratio = 0.1

        T = times[-1] - times[0]
        n = len(times)
        mu_est = (n / T) * (1 - branching_ratio) if T > 0 else self.params.mu
        beta_est = self.params.beta
        alpha_est = branching_ratio * beta_est

        try:
            return HawkesParameters(mu=float(mu_est), alpha=float(alpha_est), beta=float(beta_est))
        except ValueError:
            return None

    def _estimate_parameters_mle(self, init: HawkesParameters) -> Optional[HawkesParameters]:
        """Marked Hawkes MLE using log-likelihood minimization."""
        if not HAS_SCIPY_OPT:
            return None
        if len(self.trade_times) < 30:
            return None

        times = np.array(self.trade_times, dtype=float)
        marks = np.array(self.trade_sizes, dtype=float)
        marks = np.maximum(marks, 1e-8) ** self.mark_power
        T = max(times[-1] - times[0], 1e-8)
        shifted_times = times - times[0]

        def neg_log_likelihood(x: np.ndarray) -> float:
            mu, alpha, beta = float(x[0]), float(x[1]), float(x[2])
            if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
                return 1e12

            A = 0.0
            last_t = 0.0
            log_like = 0.0
            for t_i, mark_i in zip(shifted_times, marks):
                dt = max(0.0, float(t_i - last_t))
                A *= np.exp(-beta * dt)
                lam = mu + alpha * A
                if lam <= 1e-12 or not np.isfinite(lam):
                    return 1e12
                log_like += np.log(lam)
                A += mark_i
                last_t = float(t_i)

            integral = mu * T + (alpha / beta) * np.sum(marks * (1.0 - np.exp(-beta * (T - shifted_times))))
            nll = integral - log_like
            if not np.isfinite(nll):
                return 1e12
            return float(nll)

        x0 = np.array([init.mu, init.alpha, init.beta], dtype=float)
        bounds = [(1e-5, 20.0), (1e-5, 10.0), (1e-4, 20.0)]
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            return None
        mu, alpha, beta = map(float, result.x)
        alpha = min(alpha, beta * 0.95)

        try:
            return HawkesParameters(mu=mu, alpha=alpha, beta=beta)
        except ValueError:
            return None

    def estimate_parameters_online(self, use_mle: bool = False) -> Optional[HawkesParameters]:
        """Update Hawkes parameters using recent trade history.

        Uses method of moments for fast online estimation.

        Returns:
            Updated parameters or None if insufficient data
        """
        moments_est = self._estimate_parameters_moments()
        if moments_est is None:
            return None
        if not use_mle:
            return moments_est

        mle_est = self._estimate_parameters_mle(moments_est)
        return mle_est if mle_est is not None else moments_est


class HawkesMarketMaker(MarketMakingStrategy):
    """Market making strategy with Hawkes process intensity monitoring.

    Key innovations:
    1. Dynamic spread based on predicted trade intensity
    2. Activity-aware inventory management
    3. Adverse selection detection and avoidance
    """

    def __init__(self, config: Optional[HawkesMMConfig] = None):
        """Initialize Hawkes market maker.

        Args:
            config: Configuration. If None, uses defaults from environment variables.
        """
        self.config = config or HawkesMMConfig()
        self.name = "HawkesMM"

        # Initialize Hawkes intensity monitor
        hawkes_params = HawkesParameters(
            mu=self.config.hawkes_mu,
            alpha=self.config.hawkes_alpha,
            beta=self.config.hawkes_beta
        )
        self.monitor = HawkesIntensityMonitor(
            params=hawkes_params,
            window_size=self.config.estimation_window,
            mark_power=self.config.mark_power,
        )

        # State tracking
        self.last_trade_time: Optional[float] = None
        self.last_price: Optional[float] = None
        self.trade_count: int = 0

    def _compute_dynamic_spread(self, intensity: float, inventory: float) -> float:
        """Compute dynamic spread based on market intensity and inventory.

        Strategy:
        - High intensity -> narrow spread (capture more trades)
        - Low intensity -> widen spread (compensate for lower fill probability)
        - Large inventory -> widen spread on that side (risk management)

        Args:
            intensity: Current Hawkes intensity λ(t)
            inventory: Current position (positive = long, negative = short)

        Returns:
            Spread in basis points
        """
        # Base adjustment based on intensity relative to long-term average
        long_term = self.monitor.params.long_term_intensity
        intensity_ratio = intensity / long_term if long_term > 0 else 1.0

        # Inverse relationship: high intensity -> narrow spread
        if intensity_ratio > 2.0:  # Very active
            spread = self.config.base_spread_bps * self.config.aggressive_factor
        elif intensity_ratio > 1.0:  # Above average
            spread = self.config.base_spread_bps * (1.0 - 0.3 * (intensity_ratio - 1.0))
        elif intensity_ratio > 0.5:  # Below average
            spread = self.config.base_spread_bps * (1.0 + 0.5 * (1.0 - intensity_ratio))
        else:  # Quiet
            spread = self.config.base_spread_bps * self.config.defensive_factor

        # Inventory-based adjustment
        inventory_ratio = abs(inventory) / self.config.inventory_limit
        if inventory_ratio > 0.5:
            # Widen spread as we accumulate inventory
            spread *= (1.0 + self.config.inventory_skew_factor * inventory_ratio)

        return np.clip(spread, self.config.min_spread_bps, self.config.max_spread_bps)

    def _compute_inventory_skew(self, inventory: float, buy_int: float, sell_int: float) -> float:
        """Compute quote skew based on inventory and order flow imbalance.

        Args:
            inventory: Current position
            buy_int: Buy trade intensity
            sell_int: Sell trade intensity

        Returns:
            Skew factor (-1 to 1, negative = skew toward selling)
        """
        # Inventory skew: want to reduce large positions
        inventory_skew = -np.sign(inventory) * min(abs(inventory) / self.config.inventory_limit, 1.0)

        # Flow skew: anticipate direction from trade imbalance
        total_int = buy_int + sell_int
        if total_int > 0:
            flow_imbalance = (buy_int - sell_int) / total_int  # -1 to 1
            flow_skew = flow_imbalance * 0.3  # Moderate weight
        else:
            flow_skew = 0.0

        # Combine (inventory takes priority)
        combined_skew = 0.7 * inventory_skew + 0.3 * flow_skew
        return np.clip(combined_skew, -1.0, 1.0)

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Generate quotes based on Hawkes intensity and market state.

        Args:
            state: Current market state including order book
            position: Current position

        Returns:
            QuoteAction with bid/ask prices and sizes
        """
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")

        current_time = state.timestamp.timestamp() if hasattr(state.timestamp, 'timestamp') else 0.0

        # Update monitor with recent trades from market data if available
        if hasattr(state, 'recent_trades') and state.recent_trades:
            for trade in state.recent_trades:
                if trade.timestamp != self.last_trade_time:
                    side_val = trade.side.value if hasattr(trade.side, "value") else trade.side
                    direction = 1 if side_val == 'buy' else -1 if side_val == 'sell' else 0
                    trade_ts = trade.timestamp.timestamp() if hasattr(trade.timestamp, "timestamp") else float(trade.timestamp)
                    self.monitor.add_trade(trade_ts, direction, size=float(getattr(trade, "size", 1.0)))
                    self.last_trade_time = trade.timestamp
                    self.trade_count += 1

        # Compute current intensity
        intensity = self.monitor.get_intensity(current_time)
        buy_int, sell_int = self.monitor.get_buy_sell_intensity(current_time)

        # Check for adverse selection
        price_change = 0.0
        if self.last_price is not None:
            price_change = (mid - self.last_price) / self.last_price if self.last_price > 0 else 0.0

        adverse_selection = self.monitor.detect_adverse_selection(current_time, price_change)

        # Adjust spread based on conditions
        if adverse_selection:
            # Be defensive: widen spread significantly
            spread_bps = self.config.max_spread_bps * 0.8
        else:
            spread_bps = self._compute_dynamic_spread(intensity, position.size)

        # Compute inventory skew
        skew = self._compute_inventory_skew(position.size, buy_int, sell_int)
        flow_imbalance = 0.0
        if buy_int + sell_int > 0:
            flow_imbalance = (buy_int - sell_int) / (buy_int + sell_int)

        # Apply skew to mid price
        max_skew_bps = spread_bps * 0.3  # Max 30% of spread can be skew
        skew_bps = skew * max_skew_bps
        adjusted_mid = mid * (1 + skew_bps / 10000)

        # Calculate final prices
        half_spread = mid * spread_bps / 10000 / 2

        # Asymmetric adverse selection response:
        # if buy intensity dominates, getting lifted on ask is riskier -> widen ask more.
        # if sell intensity dominates, getting hit on bid is riskier -> widen bid more.
        imbalance = 0.0
        total_int = buy_int + sell_int
        if total_int > 0:
            imbalance = (buy_int - sell_int) / total_int
        asym_factor = np.clip(abs(imbalance), 0.0, 1.0) * 0.5

        bid_half = half_spread
        ask_half = half_spread
        if adverse_selection:
            if imbalance > 0:
                ask_half *= (1.0 + asym_factor)
                bid_half *= (1.0 - 0.3 * asym_factor)
            elif imbalance < 0:
                bid_half *= (1.0 + asym_factor)
                ask_half *= (1.0 - 0.3 * asym_factor)

        bid_price = adjusted_mid - bid_half
        ask_price = adjusted_mid + ask_half

        # Adjust sizes based on intensity
        if intensity > self.config.high_intensity_threshold:
            # High activity: quote more aggressively
            bid_size = self.config.quote_size * 1.2
            ask_size = self.config.quote_size * 1.2
        elif intensity < self.config.low_intensity_threshold:
            # Low activity: reduce exposure
            bid_size = self.config.quote_size * 0.8
            ask_size = self.config.quote_size * 0.8
        else:
            bid_size = self.config.quote_size
            ask_size = self.config.quote_size

        # Inventory-based size adjustment
        if position.size > self.config.inventory_limit * 0.5:
            # Long inventory: reduce bid, increase ask
            bid_size *= 0.7
            ask_size *= 1.3
        elif position.size < -self.config.inventory_limit * 0.5:
            # Short inventory: increase bid, reduce ask
            bid_size *= 1.3
            ask_size *= 0.7

        # Reduce risky side size under adverse selection.
        if adverse_selection:
            if imbalance > 0:
                ask_size *= 0.7
            elif imbalance < 0:
                bid_size *= 0.7

        self.last_price = mid

        control_signals = {
            "intensity": float(intensity),
            "buy_intensity": float(buy_int),
            "sell_intensity": float(sell_int),
            "flow_imbalance": float(flow_imbalance),
            "adverse_selection": bool(adverse_selection),
            "spread_bps": float(spread_bps),
            "skew_signal": float(skew),
        }

        return QuoteAction(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            metadata={
                "strategy": self.name,
                "control_signals": control_signals,
                "hawkes_params": {
                    "mu": float(self.monitor.params.mu),
                    "alpha": float(self.monitor.params.alpha),
                    "beta": float(self.monitor.params.beta),
                },
            },
        )

    def get_internal_state(self) -> Dict:
        """Return current internal state for debugging and logging."""
        return {
            'name': self.name,
            'trade_count': self.trade_count,
            'hawkes_params': {
                'mu': self.monitor.params.mu,
                'alpha': self.monitor.params.alpha,
                'beta': self.monitor.params.beta,
                'branching_ratio': self.monitor.params.branching_ratio,
                'long_term_intensity': self.monitor.params.long_term_intensity
            },
            'recent_trades_in_window': len(self.monitor.trade_times),
            'last_price': self.last_price,
            'last_trade_time': self.last_trade_time,
            'config': {
                'base_spread_bps': self.config.base_spread_bps,
                'min_spread_bps': self.config.min_spread_bps,
                'max_spread_bps': self.config.max_spread_bps,
                'quote_size': self.config.quote_size,
                'inventory_limit': self.config.inventory_limit
            }
        }

    def get_status(self) -> Dict:
        """Get current strategy status for monitoring (alias for get_internal_state)."""
        return self.get_internal_state()


class AdaptiveHawkesMarketMaker(HawkesMarketMaker):
    """Advanced version with online parameter estimation.

    This strategy continuously updates Hawkes parameters based on
    recent market activity, adapting to changing market regimes.
    """

    def __init__(self, config: Optional[HawkesMMConfig] = None):
        """Initialize adaptive Hawkes market maker."""
        super().__init__(config)
        self.name = "AdaptiveHawkesMM"
        self.param_update_interval = 50  # Update params every 50 trades
        self.last_param_update = 0

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Generate quotes with online parameter adaptation."""
        # Update parameters periodically
        if self.trade_count - self.last_param_update >= self.param_update_interval:
            new_params = self.monitor.estimate_parameters_online(use_mle=self.config.enable_online_mle)
            if new_params is not None:
                self.monitor.params = new_params
                self.last_param_update = self.trade_count
                print(f"Updated Hawkes params: μ={new_params.mu:.3f}, "
                      f"α={new_params.alpha:.3f}, β={new_params.beta:.3f}")

        # Call parent method
        return super().quote(state, position)
