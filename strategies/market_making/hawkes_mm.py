"""Hawkes-process market-making strategy."""
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
from strategies.market_making.hawkes_quote_helpers import (
    build_control_signals as _build_control_signals,
    build_quote_metadata as _build_quote_metadata,
    price_change_ratio as _price_change_ratio,
    select_spread_bps as _select_spread_bps,
    timestamp_seconds as _timestamp_seconds,
)


def _flow_imbalance(buy_intensity: float, sell_intensity: float) -> float:
    total = buy_intensity + sell_intensity
    if total <= 0:
        return 0.0
    return float((buy_intensity - sell_intensity) / total)


def _build_hawkes_quote_action(
    *,
    strategy_name: str,
    quote: Tuple[float, float, float, float],
    control_signals: Dict[str, float],
    hawkes_params: Tuple[float, float, float],
) -> QuoteAction:
    bid_price, bid_size, ask_price, ask_size = quote
    mu, alpha, beta = hawkes_params
    return QuoteAction(
        bid_price=bid_price,
        bid_size=bid_size,
        ask_price=ask_price,
        ask_size=ask_size,
        metadata=_build_quote_metadata(
            strategy_name=strategy_name,
            control_signals=control_signals,
            mu=mu,
            alpha=alpha,
            beta=beta,
        ),
    )


def _is_valid_hawkes_candidate(mu: float, alpha: float, beta: float) -> bool:
    return mu > 0 and alpha >= 0 and beta > 0 and alpha < beta


def _marked_hawkes_nll(
    x: np.ndarray,
    shifted_times: np.ndarray,
    marks: np.ndarray,
    horizon: float,
) -> float:
    mu, alpha, beta = map(float, x)
    if not _is_valid_hawkes_candidate(mu, alpha, beta):
        return 1e12
    A, last_t, log_like = 0.0, 0.0, 0.0
    for t_i, mark_i in zip(shifted_times, marks):
        dt = max(0.0, float(t_i - last_t))
        A *= np.exp(-beta * dt)
        lam = mu + alpha * A
        if lam <= 1e-12 or not np.isfinite(lam):
            return 1e12
        log_like += np.log(lam)
        A += float(mark_i)
        last_t = float(t_i)
    integral = mu * horizon + (alpha / beta) * np.sum(marks * (1.0 - np.exp(-beta * (horizon - shifted_times))))
    nll = integral - log_like
    if not np.isfinite(nll):
        return 1e12
    return float(nll)


def _optimize_marked_hawkes_params(
    init: HawkesParameters,
    shifted_times: np.ndarray,
    marks: np.ndarray,
    horizon: float,
) -> Optional[np.ndarray]:
    x0 = np.array([init.mu, init.alpha, init.beta], dtype=float)
    bounds = [(1e-5, 20.0), (1e-5, 10.0), (1e-4, 20.0)]
    result = minimize(
        _marked_hawkes_nll,
        x0,
        args=(shifted_times, marks, horizon),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        return None
    return np.array(result.x, dtype=float)


def _intensity_size_multiplier(
    intensity: float, low_threshold: float, high_threshold: float
) -> float:
    if intensity > high_threshold:
        return 1.2
    if intensity < low_threshold:
        return 0.8
    return 1.0


def _inventory_size_bias(inventory: float, inventory_limit: float) -> Tuple[float, float]:
    half_limit = inventory_limit * 0.5
    if inventory > half_limit:
        return 0.7, 1.3
    if inventory < -half_limit:
        return 1.3, 0.7
    return 1.0, 1.0


def _adverse_selection_size_bias(adverse_selection: bool, imbalance: float) -> Tuple[float, float]:
    if not adverse_selection:
        return 1.0, 1.0
    if imbalance > 0:
        return 1.0, 0.7
    if imbalance < 0:
        return 0.7, 1.0
    return 1.0, 1.0


def _validated_hawkes_parameters(mu: float, alpha: float, beta: float) -> Optional[HawkesParameters]:
    try:
        return HawkesParameters(mu=float(mu), alpha=float(alpha), beta=float(beta))
    except ValueError:
        return None


@dataclass
class HawkesMMConfig:
    """Configuration for Hawkes-based market maker."""
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
    """Monitor and estimate Hawkes intensity from recent trade flow."""

    def __init__(self, params: HawkesParameters, window_size: int = 100, mark_power: float = 0.5):
        """Initialize intensity monitor."""
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
        """Record a new trade and update O(1) recursive intensity."""
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
        """Compute conditional intensity λ(t) at current time in O(1)."""
        if self._last_event_time <= 0 or not self.trade_times:
            return self.params.mu

        dt = current_time - self._last_event_time
        A_now = np.exp(-self.params.beta * max(0.0, dt)) * self._A
        return self.params.mu + A_now

    def get_buy_sell_intensity(self, current_time: float) -> Tuple[float, float]:
        """Compute separate intensities for buy and sell trades."""
        if self._last_event_time <= 0 or not self.trade_times:
            return self.params.mu / 2, self.params.mu / 2

        dt = max(0.0, current_time - self._last_event_time)
        decay = np.exp(-self.params.beta * dt)
        buy_intensity = self.params.mu / 2 + decay * self._A_buy
        sell_intensity = self.params.mu / 2 + decay * self._A_sell
        return float(buy_intensity), float(sell_intensity)

    def detect_adverse_selection(self, current_time: float, price_change: float) -> bool:
        """Detect potential adverse selection."""
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
        branching_ratio = min(0.8, (cv - 1) / cv) if cv > 1.2 else 0.1
        T = times[-1] - times[0]
        mu_est = (len(times) / T) * (1 - branching_ratio) if T > 0 else self.params.mu
        beta_est = self.params.beta
        alpha_est = branching_ratio * beta_est
        return _validated_hawkes_parameters(mu_est, alpha_est, beta_est)

    def _estimate_parameters_mle(self, init: HawkesParameters) -> Optional[HawkesParameters]:
        """Marked Hawkes MLE using log-likelihood minimization."""
        if not HAS_SCIPY_OPT or len(self.trade_times) < 30:
            return None
        times = np.array(self.trade_times, dtype=float)
        marks = np.maximum(np.array(self.trade_sizes, dtype=float), 1e-8) ** self.mark_power
        horizon = max(times[-1] - times[0], 1e-8)
        shifted_times = times - times[0]

        mle_solution = _optimize_marked_hawkes_params(init, shifted_times, marks, horizon)
        if mle_solution is None:
            return None

        mu, alpha, beta = map(float, mle_solution)
        alpha = min(alpha, beta * 0.95)
        return _validated_hawkes_parameters(mu, alpha, beta)

    def estimate_parameters_online(self, use_mle: bool = False) -> Optional[HawkesParameters]:
        """Update Hawkes parameters using recent trade history."""
        moments_est = self._estimate_parameters_moments()
        if moments_est is None:
            return None
        if not use_mle:
            return moments_est

        mle_est = self._estimate_parameters_mle(moments_est)
        return mle_est if mle_est is not None else moments_est


class HawkesMarketMaker(MarketMakingStrategy):
    """Market-making strategy with Hawkes intensity monitoring."""

    def __init__(self, config: Optional[HawkesMMConfig] = None):
        """Initialize Hawkes market maker."""
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
        self._seen_trade_keys: set = set()
        self._seen_trade_order: deque = deque()
        self._seen_trade_key_limit: int = max(2000, self.config.estimation_window * 50)

    def _build_trade_key(self, trade) -> Tuple:
        """Build a stable key for deduplicating trade ingestion across quote calls."""
        trade_id = getattr(trade, "trade_id", None)
        if trade_id not in (None, ""):
            return ("id", str(trade_id))

        ts = trade.timestamp.timestamp() if hasattr(trade.timestamp, "timestamp") else float(trade.timestamp)
        side_val = trade.side.value if hasattr(trade.side, "value") else trade.side
        return (
            "fallback",
            str(getattr(trade, "instrument", "")),
            float(ts),
            float(getattr(trade, "price", 0.0)),
            float(getattr(trade, "size", 1.0)),
            str(side_val),
        )

    def _remember_trade_key(self, key: Tuple) -> None:
        """Track recently ingested trades with bounded memory."""
        if key in self._seen_trade_keys:
            return
        if len(self._seen_trade_order) >= self._seen_trade_key_limit:
            oldest = self._seen_trade_order.popleft()
            self._seen_trade_keys.discard(oldest)
        self._seen_trade_order.append(key)
        self._seen_trade_keys.add(key)

    def _compute_dynamic_spread(self, intensity: float, inventory: float) -> float:
        """Compute dynamic spread based on market intensity and inventory."""
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
        """Compute quote skew based on inventory and order flow imbalance."""
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

    def _ingest_recent_trades(self, state: MarketState) -> None:
        """Ingest unseen recent trades into Hawkes monitor."""
        if not hasattr(state, "recent_trades") or not state.recent_trades:
            return

        for trade in state.recent_trades:
            trade_key = self._build_trade_key(trade)
            if trade_key in self._seen_trade_keys:
                continue
            side_val = trade.side.value if hasattr(trade.side, "value") else trade.side
            direction = 1 if side_val == "buy" else -1 if side_val == "sell" else 0
            trade_ts = (
                trade.timestamp.timestamp()
                if hasattr(trade.timestamp, "timestamp")
                else float(trade.timestamp)
            )
            self.monitor.add_trade(trade_ts, direction, size=float(getattr(trade, "size", 1.0)))
            self._remember_trade_key(trade_key)
            self.last_trade_time = trade.timestamp
            self.trade_count += 1

    def _compute_quote_prices(
        self,
        *,
        mid: float,
        spread_bps: float,
        skew: float,
        buy_int: float,
        sell_int: float,
        adverse_selection: bool,
    ) -> Tuple[float, float, float, float]:
        """Compute final bid/ask with skew and adverse-selection asymmetry."""
        max_skew_bps = spread_bps * 0.3  # Max 30% of spread can be skew
        skew_bps = skew * max_skew_bps
        adjusted_mid = mid * (1 + skew_bps / 10000)
        half_spread = mid * spread_bps / 10000 / 2
        imbalance = _flow_imbalance(buy_int, sell_int)
        asym_factor = np.clip(abs(imbalance), 0.0, 1.0) * 0.5
        bid_half = half_spread
        ask_half = half_spread
        if adverse_selection and imbalance != 0:
            widen = 1.0 + asym_factor
            tighten = 1.0 - 0.3 * asym_factor
            bid_mult, ask_mult = (tighten, widen) if imbalance > 0 else (widen, tighten)
            bid_half *= bid_mult
            ask_half *= ask_mult
        return adjusted_mid - bid_half, adjusted_mid + ask_half, imbalance, adjusted_mid

    def _compute_quote_sizes(
        self,
        *,
        intensity: float,
        inventory: float,
        adverse_selection: bool,
        imbalance: float,
    ) -> Tuple[float, float]:
        """Compute quote sizes with activity, inventory, and adverse-selection controls."""
        base_multiplier = _intensity_size_multiplier(
            intensity=intensity,
            low_threshold=self.config.low_intensity_threshold,
            high_threshold=self.config.high_intensity_threshold,
        )
        bid_size = self.config.quote_size * base_multiplier
        ask_size = self.config.quote_size * base_multiplier

        inv_bid_mult, inv_ask_mult = _inventory_size_bias(
            inventory=inventory,
            inventory_limit=self.config.inventory_limit,
        )
        bid_size *= inv_bid_mult
        ask_size *= inv_ask_mult

        adv_bid_mult, adv_ask_mult = _adverse_selection_size_bias(
            adverse_selection=adverse_selection,
            imbalance=imbalance,
        )
        bid_size *= adv_bid_mult
        ask_size *= adv_ask_mult
        return bid_size, ask_size

    def _compute_quote_context(
        self,
        *,
        state: MarketState,
        mid: float,
        inventory: float,
        current_time: float,
    ) -> Tuple[float, float, float, bool, float, float, float]:
        self._ingest_recent_trades(state)
        intensity = self.monitor.get_intensity(current_time)
        buy_int, sell_int = self.monitor.get_buy_sell_intensity(current_time)
        price_change = _price_change_ratio(mid, self.last_price)
        adverse_selection = self.monitor.detect_adverse_selection(current_time, price_change)
        spread_bps = _select_spread_bps(
            adverse_selection=adverse_selection,
            max_spread_bps=self.config.max_spread_bps,
            dynamic_spread_bps=self._compute_dynamic_spread(intensity, inventory),
        )
        skew = self._compute_inventory_skew(inventory, buy_int, sell_int)
        flow_imbalance = _flow_imbalance(buy_int, sell_int)
        return intensity, buy_int, sell_int, adverse_selection, spread_bps, skew, flow_imbalance

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Generate quotes based on Hawkes intensity and market state."""
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")
        current_time = _timestamp_seconds(state.timestamp)
        intensity, buy_int, sell_int, adverse_selection, spread_bps, skew, flow_imbalance = self._compute_quote_context(state=state, mid=mid, inventory=position.size, current_time=current_time)
        bid_price, ask_price, imbalance, _ = self._compute_quote_prices(
            mid=mid,
            spread_bps=spread_bps,
            skew=skew,
            buy_int=buy_int,
            sell_int=sell_int,
            adverse_selection=adverse_selection,
        )
        bid_size, ask_size = self._compute_quote_sizes(intensity=intensity, inventory=position.size, adverse_selection=adverse_selection, imbalance=imbalance)
        control_signals = _build_control_signals(intensity=float(intensity), buy_intensity=float(buy_int), sell_intensity=float(sell_int), flow_imbalance=float(flow_imbalance), adverse_selection=bool(adverse_selection), spread_bps=float(spread_bps), skew=float(skew))
        self.last_price = mid
        return _build_hawkes_quote_action(strategy_name=self.name, quote=(bid_price, bid_size, ask_price, ask_size), control_signals=control_signals, hawkes_params=(self.monitor.params.mu, self.monitor.params.alpha, self.monitor.params.beta))

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
    """Hawkes market maker with online parameter estimation."""

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
