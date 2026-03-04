"""Integrated market-making strategy with risk controls and hedging."""
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.types import Greeks, MarketState, Position, QuoteAction
from research.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    PortfolioState,
    TradeAction,
)
from research.signals.regime_detector import (
    RegimeConfig,
    RegimeState,
    VolatilityRegimeDetector,
)
from research.hedging.adaptive_delta import (
    AdaptiveDeltaHedger,
    AdaptiveHedgeConfig,
    HedgeDecision,
)
from strategies.base import MarketMakingStrategy


def _log_return(current_price: float, prev_price: Optional[float]) -> Optional[float]:
    if prev_price is None or prev_price <= 0:
        return None
    return float(np.log(current_price / prev_price))


@dataclass
class IntegratedStrategyConfig:
    """Configuration for integrated market making strategy."""

    # Sub-component configurations
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    regime_detector: RegimeConfig = field(default_factory=RegimeConfig)
    adaptive_hedger: AdaptiveHedgeConfig = field(default_factory=AdaptiveHedgeConfig)

    # Base market making parameters
    base_spread_bps: float = 20.0
    quote_size: float = 1.0
    inventory_limit: float = 10.0

    # AS model parameters (for reservation price)
    gamma: float = 0.1  # Risk aversion
    sigma: float = 0.5  # Base volatility assumption

    # Inventory skew parameters
    inventory_skew_factor: float = 0.5  # How much to skew quotes
    max_skew_bps: float = 50.0  # Maximum skew in basis points

    # Spread adjustment per regime (multipliers on base_spread_bps)
    regime_spread_multipliers: Dict[RegimeState, float] = field(default_factory=lambda: {
        RegimeState.LOW: 0.8,
        RegimeState.MEDIUM: 1.0,
        RegimeState.HIGH: 1.5
    })

    # Hedging parameters
    enable_adaptive_hedging: bool = True
    target_delta: float = 0.0  # Delta-neutral target

    # PnL tracking
    initial_capital: float = 1.0
    max_pnl_history: int = 2000

    # Online calibration
    enable_online_calibration: bool = False
    calibration_window: int = 60
    min_sigma: float = 0.05
    max_sigma: float = 2.0
    min_inventory_skew_factor: float = 0.05
    max_inventory_skew_factor: float = 2.0


@dataclass
class StrategyMetrics:
    """Metrics tracked by the strategy."""
    timestamp: datetime
    regime: RegimeState
    circuit_state: str
    spread_bps: float
    inventory: float
    delta: float
    pnl: float


class IntegratedMarketMakingStrategy(MarketMakingStrategy):
    """Integrated market-making strategy with risk management."""

    def __init__(self, config: IntegratedStrategyConfig = None):
        self.config = config or IntegratedStrategyConfig()
        self.name = "IntegratedMarketMaking"

        # Initialize sub-components
        self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker)
        self.regime_detector = VolatilityRegimeDetector(self.config.regime_detector)
        self.hedger = AdaptiveDeltaHedger(self.config.adaptive_hedger)

        # State tracking
        self._returns_history: List[float] = []
        self._pnl_history: Deque[Tuple[datetime, float]] = deque(maxlen=self.config.max_pnl_history)
        self._metrics_history: List[StrategyMetrics] = []
        self._current_greeks: Optional[Greeks] = None
        self._pnl_series_cache: pd.Series = pd.Series(dtype=float)
        self._trade_intensity_history: Deque[float] = deque(maxlen=max(5, self.config.calibration_window))
        self._effective_sigma: float = float(self.config.sigma)
        self._effective_inventory_skew_factor: float = float(self.config.inventory_skew_factor)

        # Portfolio tracking
        self._current_position: Optional[Position] = None
        self._current_price: float = 0.0
        self._realized_pnl: float = 0.0

    def _update_online_calibration(self, state: MarketState, inventory: float) -> Dict[str, float]:
        """Adapt sigma and inventory skew from rolling returns and flow intensity."""
        if not self.config.enable_online_calibration:
            self._effective_sigma = float(self.config.sigma)
            self._effective_inventory_skew_factor = float(self.config.inventory_skew_factor)
            return {}

        intensity_feature = state.features.get("trade_intensity")
        intensity = float(max(0.0, intensity_feature)) if intensity_feature is not None else float(len(state.recent_trades))
        self._trade_intensity_history.append(intensity)

        if len(self._returns_history) >= 5:
            sigma_raw = float(np.std(self._returns_history[-self.config.calibration_window:], ddof=1)) * np.sqrt(365.25 * 24 * 3600)
            if np.isfinite(sigma_raw):
                self._effective_sigma = float(np.clip(sigma_raw, self.config.min_sigma, self.config.max_sigma))

        if len(self._trade_intensity_history) >= 3:
            avg_intensity = float(np.mean(self._trade_intensity_history))
            inventory_util = abs(float(inventory)) / max(self.config.inventory_limit, 1e-12)
            # Higher inventory or flow intensity increases skew pressure.
            skew_raw = self.config.inventory_skew_factor * (1.0 + 0.2 * avg_intensity + 0.5 * inventory_util)
            self._effective_inventory_skew_factor = float(
                np.clip(skew_raw, self.config.min_inventory_skew_factor, self.config.max_inventory_skew_factor)
            )

        return {
            "calibrated_sigma": self._effective_sigma,
            "calibrated_inventory_skew_factor": self._effective_inventory_skew_factor,
        }

    def _update_regime_state(self, mid: float, prev_price: Optional[float]) -> RegimeState:
        """Update returns/regime from latest mid price."""
        ret = _log_return(mid, prev_price)
        if ret is not None:
            detector_input = np.expm1(ret) if self.regime_detector.config.use_log_returns else ret
            self.regime_detector.update(detector_input)
            self._returns_history.append(ret)
        return self.regime_detector.current_regime

    def _evaluate_trading_gate(
        self, state: MarketState, position: Position
    ) -> Tuple[object, bool, str]:
        """Run circuit-breaker risk checks and trading gate evaluation."""
        portfolio = self._update_portfolio_state(state, position)
        circuit_state = self.circuit_breaker.check_risk_limits(portfolio)
        can_trade, reason = self.circuit_breaker.can_trade(TradeAction.MARKET_MAKING)
        return circuit_state, can_trade, reason

    def _evaluate_hedge_decision(
        self, state: MarketState, mid: float, position: Position
    ) -> Optional[HedgeDecision]:
        """Evaluate and execute adaptive hedge when enabled and required."""
        if not self.config.enable_adaptive_hedging or self._current_greeks is None:
            return None
        hedge_decision = self.hedger.should_hedge(
            state.timestamp,
            mid,
            self._current_greeks,
            position.size,
        )
        if hedge_decision.should_hedge:
            self.hedger.execute_hedge(state.timestamp, hedge_decision.hedge_size, mid)
        return hedge_decision

    def _build_quote_metadata(
        self,
        *,
        circuit_state: object,
        current_regime: RegimeState,
        spread_bps: float,
        spread_multiplier: float,
        reservation_price: float,
        half_spread: float,
        position: Position,
        effective_inventory_limit: float,
        position_limit_mult: float,
        hedge_decision: Optional[HedgeDecision],
        calibration_meta: Dict[str, float],
    ) -> Dict[str, object]:
        """Build standardized quote metadata for observability."""
        return {
            "strategy": self.name,
            "circuit_state": circuit_state.value,
            "regime": current_regime.name,
            "spread_bps": spread_bps,
            "spread_multiplier": spread_multiplier,
            "reservation_price": reservation_price,
            "half_spread": half_spread,
            "inventory": position.size,
            "inventory_limit": effective_inventory_limit,
            "position_limit_multiplier": position_limit_mult,
            "hedge_decision": hedge_decision.reason if hedge_decision else None,
            "hedge_urgency": hedge_decision.urgency if hedge_decision else None,
            "regime_switch_prob": self.regime_detector.predict_regime_switch_probability(),
            "circuit_breaker_triggers": len(self.circuit_breaker.violation_history),
            **calibration_meta,
        }

    def _prepare_quote_state(
        self, state: MarketState, position: Position
    ) -> Tuple[float, RegimeState, Dict[str, float], object, bool, str]:
        """Prepare shared quote state before branching on trade gate."""
        self._current_position = position
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")
        prev_price = self._current_price if self._current_price > 0 else None
        self._current_price = mid
        self._current_greeks = state.greeks

        current_regime = self._update_regime_state(mid, prev_price)
        calibration_meta = self._update_online_calibration(state, position.size)
        circuit_state, can_trade, reason = self._evaluate_trading_gate(state, position)
        return float(mid), current_regime, calibration_meta, circuit_state, can_trade, reason

    def _build_halted_quote(
        self,
        *,
        mid: float,
        circuit_state: object,
        current_regime: RegimeState,
        reason: str,
        calibration_meta: Dict[str, float],
    ) -> QuoteAction:
        """Build zero-size quote when risk gate blocks trading."""
        return QuoteAction(
            bid_price=mid,
            bid_size=0.0,
            ask_price=mid,
            ask_size=0.0,
            metadata={
                "strategy": self.name,
                "circuit_state": circuit_state.value,
                "regime": current_regime.name,
                "trading_halted": True,
                "halt_reason": reason,
                **calibration_meta,
            },
        )

    def _build_active_quote(
        self,
        *,
        state: MarketState,
        position: Position,
        mid: float,
        circuit_state: object,
        current_regime: RegimeState,
        calibration_meta: Dict[str, float],
    ) -> QuoteAction:
        """Build normal market-making quote after risk gate passes."""
        hedge_decision = self._evaluate_hedge_decision(state, mid, position)
        spread_multiplier = self._get_spread_multiplier()
        spread_bps = self.config.base_spread_bps * spread_multiplier
        reservation_price, half_spread = self._calculate_reservation_price(
            mid, position.size, spread_bps
        )
        position_limit_mult = self.circuit_breaker.get_position_limit_multiplier()
        effective_inventory_limit = self.config.inventory_limit * position_limit_mult
        bid_size, ask_size = self._calculate_quote_sizes(position.size, effective_inventory_limit)
        metadata = self._build_quote_metadata(
            circuit_state=circuit_state,
            current_regime=current_regime,
            spread_bps=spread_bps,
            spread_multiplier=spread_multiplier,
            reservation_price=reservation_price,
            half_spread=half_spread,
            position=position,
            effective_inventory_limit=effective_inventory_limit,
            position_limit_mult=position_limit_mult,
            hedge_decision=hedge_decision,
            calibration_meta=calibration_meta,
        )
        self._record_metrics(
            state.timestamp, current_regime, circuit_state.value, spread_bps, position.size
        )
        return QuoteAction(
            bid_price=reservation_price - half_spread,
            bid_size=bid_size,
            ask_price=reservation_price + half_spread,
            ask_size=ask_size,
            metadata=metadata,
        )

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Generate quotes using integrated strategy."""
        mid, current_regime, calibration_meta, circuit_state, can_trade, reason = (
            self._prepare_quote_state(state, position)
        )
        if not can_trade:
            return self._build_halted_quote(
                mid=mid,
                circuit_state=circuit_state,
                current_regime=current_regime,
                reason=reason,
                calibration_meta=calibration_meta,
            )
        return self._build_active_quote(
            state=state,
            position=position,
            mid=mid,
            circuit_state=circuit_state,
            current_regime=current_regime,
            calibration_meta=calibration_meta,
        )

    def _update_portfolio_state(self, state: MarketState, position: Position) -> PortfolioState:
        """Update internal portfolio state for risk checks."""
        # Calculate PnL
        current_pnl = self._calculate_pnl(position, state.spot_price)

        # Update bounded history and cached series incrementally (hot path).
        self._pnl_history.append((state.timestamp, current_pnl))
        self._pnl_series_cache.loc[state.timestamp] = current_pnl
        if len(self._pnl_series_cache) > self.config.max_pnl_history:
            self._pnl_series_cache = self._pnl_series_cache.iloc[-self.config.max_pnl_history:]

        asset_returns = pd.DataFrame()
        if self._returns_history:
            tail = self._returns_history[-self.config.max_pnl_history:]
            asset_returns = pd.DataFrame({state.instrument: tail})

        return PortfolioState(
            timestamp=state.timestamp,
            positions={state.instrument: position},
            cash=self._realized_pnl,
            pnl_series=self._pnl_series_cache,
            asset_returns=asset_returns,
            initial_capital=self.config.initial_capital
        )

    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Calculate current PnL."""
        # Realized PnL from trades + unrealized from position
        unrealized = position.unrealized_pnl(current_price, inverse=True)
        return self._realized_pnl + unrealized

    def _get_spread_multiplier(self) -> float:
        """Calculate total spread multiplier from all components."""
        # Base: regime multiplier
        regime_mult = self.regime_detector.get_spread_adjustment()

        # Circuit breaker multiplier
        circuit_mult = self.circuit_breaker.get_spread_multiplier()

        # Multiplicative composition preserves both risk and regime effects.
        combined = regime_mult * circuit_mult
        return float(np.clip(combined, 0.5, 5.0))

    def _calculate_reservation_price(
        self,
        mid: float,
        inventory: float,
        spread_bps: float
    ) -> Tuple[float, float]:
        """Calculate reservation price with inventory skew."""
        gamma = self.config.gamma
        sigma = self._effective_sigma

        # Inventory skew (AS model component)
        # When long (inventory > 0), reservation price is lower -> more willing to sell
        skew_factor = self._effective_inventory_skew_factor
        temp_skew_factor = getattr(self, "_temp_inventory_skew_factor", None)
        if temp_skew_factor is not None:
            skew_factor = float(temp_skew_factor)
        inventory_risk = inventory * gamma * sigma ** 2 * skew_factor

        # Limit skew to prevent extreme quotes
        max_skew = mid * self.config.max_skew_bps / 10000
        inventory_skew = np.clip(inventory_risk, -max_skew, max_skew)

        reservation_price = mid - inventory_skew

        # Half spread in price terms
        half_spread = mid * spread_bps / 10000 / 2

        return reservation_price, half_spread

    def _calculate_quote_sizes(
        self,
        inventory: float,
        inventory_limit: float
    ) -> Tuple[float, float]:
        """Calculate bid and ask sizes with inventory limits."""
        base_size = self.config.quote_size

        # Default sizes
        bid_size = base_size
        ask_size = base_size

        # Apply inventory limits
        if inventory + base_size > inventory_limit:
            # Would exceed long limit
            bid_size = max(0, inventory_limit - inventory)

        if inventory - base_size < -inventory_limit:
            # Would exceed short limit
            ask_size = max(0, inventory + inventory_limit)

        # Note: Circuit breaker multiplier already applied to inventory_limit
        # in the calling code (effective_inventory_limit)
        return bid_size, ask_size

    def _record_metrics(
        self,
        timestamp: datetime,
        regime: RegimeState,
        circuit_state: str,
        spread_bps: float,
        inventory: float
    ) -> None:
        """Record strategy metrics."""
        delta = 0.0
        if self._current_greeks:
            delta = self._current_greeks.delta * inventory

        pnl = 0.0
        if self._pnl_history:
            pnl = self._pnl_history[-1][1]

        metrics = StrategyMetrics(
            timestamp=timestamp,
            regime=regime,
            circuit_state=circuit_state,
            spread_bps=spread_bps,
            inventory=inventory,
            delta=delta,
            pnl=pnl
        )
        self._metrics_history.append(metrics)

    def on_fill(self, fill, position: Position) -> None:
        """Handle fill and update PnL."""
        if self._current_price <= 0:
            raise ValueError("Current price must be positive before processing fills")

        # Update realized PnL
        if fill.side.value == "buy":
            self._realized_pnl -= fill.price * fill.size / self._current_price
        else:
            self._realized_pnl += fill.price * fill.size / self._current_price

    def get_internal_state(self) -> Dict:
        """Return comprehensive internal state."""
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "regime": {
                "current": self.regime_detector.current_regime.name,
                "probabilities": self.regime_detector.regime_probabilities.tolist(),
                "switch_probability": self.regime_detector.predict_regime_switch_probability()
            },
            "hedger": self.hedger.get_hedge_stats(),
            "config": {
                "base_spread_bps": self.config.base_spread_bps,
                "inventory_limit": self.config.inventory_limit,
                "gamma": self.config.gamma
            },
            "metrics_count": len(self._metrics_history),
            "pnl": self._realized_pnl if self._pnl_history else 0.0
        }

    def reset(self) -> None:
        """Reset all strategy state."""
        self.circuit_breaker.manual_reset("Strategy reset")
        self.regime_detector.reset()
        self.hedger.reset()
        self._returns_history.clear()
        self._pnl_history.clear()
        self._pnl_series_cache = pd.Series(dtype=float)
        self._metrics_history.clear()
        self._current_greeks = None
        self._current_position = None
        self._current_price = 0.0
        self._realized_pnl = 0.0
        self._trade_intensity_history.clear()
        self._effective_sigma = float(self.config.sigma)
        self._effective_inventory_skew_factor = float(self.config.inventory_skew_factor)

    def get_metrics_df(self) -> pd.DataFrame:
        """Get strategy metrics as DataFrame."""
        if not self._metrics_history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "regime": m.regime.name,
                "circuit_state": m.circuit_state,
                "spread_bps": m.spread_bps,
                "inventory": m.inventory,
                "delta": m.delta,
                "pnl": m.pnl
            }
            for m in self._metrics_history
        ])


class IntegratedStrategyWithFeatures(IntegratedMarketMakingStrategy):
    """Extended integrated strategy with additional feature extraction."""

    def __init__(self, config: IntegratedStrategyConfig = None):
        super().__init__(config)
        self.name = "IntegratedMM_WithFeatures"
        self._temp_inventory_skew_factor: Optional[float] = None

    def _calculate_reservation_price(
        self,
        mid: float,
        inventory: float,
        spread_bps: float
    ) -> Tuple[float, float]:
        """Use temporary per-quote skew override without mutating shared config."""
        return super()._calculate_reservation_price(mid, inventory, spread_bps)

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Quote with feature extraction."""
        # Extract additional features from market state
        features = self._extract_features(state)

        # Use features to adjust strategy parameters
        if "orderbook_imbalance" in features:
            # Adjust skew based on order book imbalance
            imbalance = features["orderbook_imbalance"]
            # Positive imbalance (more bids) -> market is bullish -> skew up
            # But we want to mean-revert, so skew down (lower reservation price)
            adjusted = self.config.inventory_skew_factor * (1 - imbalance * 0.1)
            # Keep adaptive skew within a safe envelope.
            self._temp_inventory_skew_factor = float(np.clip(adjusted, 0.05, 2.0))
        else:
            self._temp_inventory_skew_factor = None

        # Call parent quote method
        try:
            quote = super().quote(state, position)
        finally:
            self._temp_inventory_skew_factor = None

        # Add features to metadata
        quote.metadata["extracted_features"] = features

        return quote

    def _extract_features(self, state: MarketState) -> Dict[str, float]:
        """Extract additional features from market state."""
        features = {}

        # Order book imbalance
        if state.order_book:
            features["orderbook_imbalance"] = state.order_book.imbalance(levels=5)

        # Spread features
        if state.order_book.spread:
            mid = state.order_book.mid_price
            if mid:
                features["spread_bps"] = (state.order_book.spread / mid) * 10000

        # Recent trade features
        if state.recent_trades:
            trade_sizes = [t.size for t in state.recent_trades]
            features["avg_trade_size"] = np.mean(trade_sizes)
            features["trade_volume"] = sum(trade_sizes)

        # Copy any pre-computed features
        features.update(state.features)

        return features
