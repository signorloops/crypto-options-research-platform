"""
Avellaneda-Stoikov market making strategy.
Classic optimal market making model from the seminal 2008 paper.
"""
from dataclasses import dataclass
from collections import deque
from typing import Dict

import numpy as np

from core.types import MarketState, Position, QuoteAction
from strategies.base import MarketMakingStrategy


@dataclass
class ASConfig:
    """Configuration for Avellaneda-Stoikov strategy."""
    gamma: float = 0.1          # Risk aversion coefficient
    sigma: float = 0.5          # Volatility (annualized)
    k: float = 1.5              # Order arrival intensity parameter
    T: float = 1.0              # Trading horizon (in years)
    quote_size: float = 1.0     # Quote size
    inventory_limit: float = 10.0  # Maximum inventory
    use_bounded_inventory: bool = True  # GLFT-style bounded inventory
    inventory_saturation: float = 0.8   # How fast inventory penalty saturates
    enable_online_calibration: bool = False
    calibration_window: int = 60
    min_sigma: float = 0.05
    max_sigma: float = 2.0
    min_k: float = 0.1
    max_k: float = 12.0


class AvellanedaStoikov(MarketMakingStrategy):
    """
    Implementation of the Avellaneda-Stoikov optimal market making model.

    Key equations:
    - Reservation price: r(s,t) = s - q * gamma * sigma^2 * (T-t)
    - Optimal spread: delta = gamma * sigma^2 * (T-t) + 2/gamma * ln(1+gamma/k)

    Where:
    - s: mid price
    - q: current inventory
    - gamma: risk aversion
    - sigma: volatility
    - T-t: time remaining
    - k: order arrival intensity

    Reference:
    Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
    Quantitative Finance, 8(3), 217-224.
    """

    def __init__(self, config: ASConfig = None):
        self.config = config or ASConfig()
        self.name = "AvellanedaStoikov"
        self._start_timestamp = None  # Track start timestamp for (T-t) calculation
        self._initial_sigma = float(self.config.sigma)
        self._initial_k = float(self.config.k)
        self._last_mid_price: float = 0.0
        window = max(5, int(self.config.calibration_window))
        self._returns_window = deque(maxlen=window)
        self._trade_intensity_window = deque(maxlen=window)

    def _update_online_calibration(self, state: MarketState, inventory: float) -> Dict[str, float]:
        """Update sigma/k online from rolling market features."""
        if not self.config.enable_online_calibration:
            return {}

        mid = state.order_book.mid_price
        if mid is None or mid <= 0:
            return {}

        if self._last_mid_price > 0:
            ret = float(np.log(mid / self._last_mid_price))
            if np.isfinite(ret):
                self._returns_window.append(ret)
        self._last_mid_price = float(mid)

        intensity_feature = state.features.get("trade_intensity")
        if intensity_feature is not None:
            intensity = float(max(0.0, intensity_feature))
        else:
            intensity = float(len(state.recent_trades))
        self._trade_intensity_window.append(intensity)

        if len(self._returns_window) >= 5:
            sigma_raw = float(np.std(self._returns_window, ddof=1)) * np.sqrt(365.25 * 24 * 3600)
            if np.isfinite(sigma_raw):
                self.config.sigma = float(np.clip(sigma_raw, self.config.min_sigma, self.config.max_sigma))

        if len(self._trade_intensity_window) >= 3:
            # k proxies fill decay against observed trade activity and inventory pressure.
            avg_intensity = float(np.mean(self._trade_intensity_window))
            inventory_util = abs(float(inventory)) / max(self.config.inventory_limit, 1e-12)
            k_raw = avg_intensity / (1.0 + 0.5 * inventory_util)
            if np.isfinite(k_raw):
                self.config.k = float(np.clip(k_raw, self.config.min_k, self.config.max_k))

        return {
            "calibrated_sigma": float(self.config.sigma),
            "calibrated_k": float(self.config.k),
        }

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """
        Generate quotes using AS model.

        The strategy adjusts quotes based on:
        1. Inventory level (skew quotes to reduce inventory)
        2. Time remaining (become more aggressive near end)
        3. Volatility (wider spreads in high vol)
        """
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")

        q = position.size  # Current inventory
        calibration_meta = self._update_online_calibration(state, q)
        gamma = self.config.gamma
        sigma = self.config.sigma
        k = self.config.k

        # Calculate actual time remaining (T-t) using market state timestamp
        # (not wall-clock time, so backtesting works correctly)
        if self._start_timestamp is None:
            self._start_timestamp = state.timestamp
        delta = state.timestamp - self._start_timestamp
        if hasattr(delta, "total_seconds"):
            elapsed_seconds = delta.total_seconds()
        else:
            # Support numpy datetime64/timedelta64 used by some backtest loaders.
            elapsed_seconds = float(delta / np.timedelta64(1, "s"))
        trading_horizon_seconds = self.config.T * 365.25 * 24 * 3600
        time_remaining = max(0, trading_horizon_seconds - elapsed_seconds) / (365.25 * 24 * 3600)

        # Reservation price (inventory-adjusted mid)
        # When long (q>0), reservation price is lower -> more willing to sell
        # When short (q<0), reservation price is higher -> more willing to buy
        inventory_ratio = q / max(self.config.inventory_limit, 1e-12)
        if self.config.use_bounded_inventory:
            # Bounded inventory transform prevents reservation price from diverging.
            bounded_ratio = np.tanh(inventory_ratio / max(self.config.inventory_saturation, 1e-6))
            effective_q = bounded_ratio * self.config.inventory_limit
        else:
            effective_q = q

        reservation_price = mid - effective_q * gamma * sigma**2 * time_remaining

        # Optimal half-spread
        # gamma * sigma^2 * T: inventory risk component
        # 2/gamma * ln(1+gamma/k): execution risk component
        spread_component = gamma * sigma**2 * time_remaining
        execution_component = (2 / gamma) * np.log(1 + gamma / k)
        # GLFT-style bounded inventory risk premium widens spread near limits.
        inventory_premium = 0.0
        if self.config.use_bounded_inventory:
            inventory_premium = spread_component * abs(inventory_ratio) ** 1.5
        half_spread = (spread_component + execution_component + inventory_premium) / 2

        # Calculate quotes
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread

        # Inventory limits
        bid_size = self.config.quote_size
        ask_size = self.config.quote_size

        if q >= self.config.inventory_limit:
            bid_size = 0  # Don't increase long position
        elif q <= -self.config.inventory_limit:
            ask_size = 0  # Don't increase short position
        else:
            # Smoothly taper size as inventory approaches limit.
            taper = max(0.0, 1.0 - abs(inventory_ratio))
            bid_size *= taper if q > 0 else 1.0
            ask_size *= taper if q < 0 else 1.0

        # Note: Reservation price already incorporates inventory adjustment
        # per Avellaneda-Stoikov model. No additional skew needed.

        return QuoteAction(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            metadata={
                "strategy": self.name,
                "mid_price": mid,
                "reservation_price": reservation_price,
                "half_spread": half_spread,
                "inventory": q,
                "effective_inventory": effective_q,
                "spread_component": spread_component,
                "execution_component": execution_component,
                "inventory_premium": inventory_premium,
                **calibration_meta,
            }
        )

    def get_internal_state(self) -> Dict:
        """Return AS-specific parameters."""
        elapsed = 0.0 if self._start_timestamp is None else 0.0
        return {
            "gamma": self.config.gamma,
            "sigma": self.config.sigma,
            "k": self.config.k,
            "elapsed_time": elapsed,
            "time_remaining": max(0, self.config.T - elapsed / (365.25 * 24 * 3600))
        }

    def reset(self) -> None:
        """Reset start timestamp."""
        self._start_timestamp = None
        self._last_mid_price = 0.0
        self._returns_window.clear()
        self._trade_intensity_window.clear()
        self.config.sigma = self._initial_sigma
        self.config.k = self._initial_k


class ASWithVolatilityAdaptation(AvellanedaStoikov):
    """
    AS model with adaptive volatility estimation.

    Instead of fixed sigma, uses recent realized volatility.
    """

    def __init__(self, config: ASConfig = None, vol_window: int = 24):
        super().__init__(config)
        self.name = "AS_AdaptiveVol"
        self.vol_window = vol_window
        self._returns_history: list = []

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Quote with adaptive volatility."""
        # Update volatility estimate from recent returns
        if "returns" in state.features:
            self._returns_history.append(state.features["returns"])
            if len(self._returns_history) > self.vol_window:
                self._returns_history.pop(0)

        # Calculate realized volatility
        if len(self._returns_history) >= 10:
            realized_vol = np.std(self._returns_history) * np.sqrt(365 * 24)  # Annualized
            self.config.sigma = max(0.1, min(2.0, realized_vol))  # Clamp between 10% and 200%

        return super().quote(state, position)

    def reset(self) -> None:
        """Clear volatility history."""
        super().reset()
        self._returns_history = []
