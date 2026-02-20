"""
高性能综合做市策略 (Fast Integrated Market Making Strategy).

针对生产环境优化的版本:
1. 使用 FastVolatilityRegimeDetector (延迟 <5ms)
2. Numba加速 Greeks计算
3. 预计算缓存
4. 异步HMM训练

性能目标:
- Quote Generation P95 < 35ms (当前108ms)
- 端到端 P95 < 10ms
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.types import Greeks, MarketState, Position, QuoteAction
from research.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    PortfolioState,
    TradeAction,
)
from research.signals.fast_regime_detector import (
    FastRegimeConfig,
    FastVolatilityRegimeDetector,
    RegimeState,
)
from research.hedging.adaptive_delta import (
    AdaptiveDeltaHedger,
    AdaptiveHedgeConfig,
)
from strategies.base import MarketMakingStrategy


@dataclass
class FastIntegratedStrategyConfig:
    """高性能策略配置."""

    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    regime_detector: FastRegimeConfig = field(default_factory=FastRegimeConfig)
    adaptive_hedger: AdaptiveHedgeConfig = field(default_factory=AdaptiveHedgeConfig)

    # 基础做市参数
    base_spread_bps: float = 20.0
    quote_size: float = 1.0
    inventory_limit: float = 10.0
    inventory_skew_factor: float = 0.5

    # 性能优化
    use_fast_regime: bool = True
    cache_greeks: bool = True
    greeks_cache_ttl_ms: float = 100.0
    greeks_cache_max_entries: int = 2048

    # 功能开关
    enable_circuit_breaker: bool = True
    enable_regime_detection: bool = True
    enable_adaptive_hedging: bool = True
    enable_inventory_skew: bool = True

    # Risk-check throttling (performance guard)
    risk_check_interval_ms: float = 250.0
    risk_check_price_move_bps: float = 20.0


class FastIntegratedMarketMakingStrategy(MarketMakingStrategy):
    """
    高性能综合做市策略.

    与标准版IntegratedMarketMakingStrategy相比:
    - Quote延迟: 108ms → <35ms (目标)
    - 使用FastRegimeDetector替代标准版
    - Greeks缓存减少重复计算
    """

    def __init__(self, config: FastIntegratedStrategyConfig = None):
        self.config = config or FastIntegratedStrategyConfig()
        self.name = "FastIntegratedMarketMaking"

        # 初始化子组件
        self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker)
        self.regime_detector = FastVolatilityRegimeDetector(self.config.regime_detector)
        self.hedger = AdaptiveDeltaHedger(self.config.adaptive_hedger)

        # 状态跟踪
        self._returns_history: List[float] = []
        self._current_greeks: Optional[Greeks] = None

        # Greeks缓存
        self._greeks_cache: OrderedDict = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # 性能统计
        self._quote_count: int = 0
        self._total_latency_ms: float = 0.0
        self._last_price: Optional[float] = None
        self._last_risk_check_at: Optional[datetime] = None
        self._last_risk_check_mid: Optional[float] = None
        self._last_risk_check_inventory: Optional[float] = None
        self._last_can_trade: bool = True
        self._last_halt_reason: str = "ok"

    def _should_refresh_risk_check(self, timestamp: datetime, mid: float, inventory: float) -> bool:
        """Decide whether to run expensive risk checks on this quote."""
        if self._last_risk_check_at is None:
            return True

        elapsed_ms = (timestamp - self._last_risk_check_at).total_seconds() * 1000
        if elapsed_ms >= self.config.risk_check_interval_ms:
            return True

        if (
            self._last_risk_check_inventory is None
            or abs(inventory - self._last_risk_check_inventory) > 1e-8
        ):
            return True

        if self._last_risk_check_mid is None or self._last_risk_check_mid <= 0:
            return True

        move_bps = abs(mid / self._last_risk_check_mid - 1.0) * 10000
        return move_bps >= self.config.risk_check_price_move_bps

    def _get_cached_greeks(self, cache_key: str) -> Optional[Greeks]:
        """获取缓存的Greeks."""
        if not self.config.cache_greeks:
            return None

        if cache_key in self._greeks_cache:
            greeks, timestamp = self._greeks_cache[cache_key]
            age_ms = (datetime.now(timezone.utc) - timestamp).total_seconds() * 1000

            if age_ms < self.config.greeks_cache_ttl_ms:
                self._cache_hits += 1
                return greeks

        self._cache_misses += 1
        return None

    def _set_cached_greeks(self, cache_key: str, greeks: Greeks) -> None:
        """设置Greeks缓存 (LRU via OrderedDict, O(1) eviction)."""
        if not self.config.cache_greeks:
            return

        now = datetime.now(timezone.utc)

        # Move to end if already exists (LRU touch)
        if cache_key in self._greeks_cache:
            del self._greeks_cache[cache_key]
        self._greeks_cache[cache_key] = (greeks, now)

        # O(1) eviction: pop from front (oldest) when over capacity
        while len(self._greeks_cache) > self.config.greeks_cache_max_entries:
            self._greeks_cache.pop(next(iter(self._greeks_cache)))

    def _update_portfolio_state(self, state: MarketState, position: Position) -> PortfolioState:
        """更新组合状态 (简化版)."""
        mid = state.order_book.mid_price or 0.0

        # 简化的PnL计算
        unrealized_pnl = 0.0
        if position.avg_entry_price > 0:
            unrealized_pnl = position.size * (1.0 / position.avg_entry_price - 1.0 / mid)

        asset_returns_df = pd.DataFrame()
        if self._returns_history:
            asset_returns_df = pd.DataFrame({position.instrument: self._returns_history[-500:]})

        return PortfolioState(
            timestamp=state.timestamp,
            positions={position.instrument: position},
            cash=unrealized_pnl,
            asset_returns=asset_returns_df,
            initial_capital=abs(position.size) * mid if position.size != 0 else 10000.0,
        )

    def _calculate_return(self, current_price: float) -> Optional[float]:
        """计算对数收益率 (与标准策略保持一致)."""
        if self._last_price is None:
            self._last_price = current_price
            return None

        if self._last_price <= 0 or current_price <= 0:
            self._last_price = current_price
            return None

        ret = np.log(current_price / self._last_price)
        self._last_price = current_price
        return ret

    def _get_spread_multiplier(self) -> float:
        """获取价差倍数."""
        multiplier = 1.0

        # Regime调整
        if self.config.enable_regime_detection:
            multiplier *= self.regime_detector.get_spread_adjustment()

        # Circuit breaker调整
        if self.config.enable_circuit_breaker:
            multiplier *= self.circuit_breaker.get_spread_multiplier()

        return max(0.5, min(multiplier, 3.0))

    def _calculate_reservation_price(
        self, mid: float, inventory: float, spread_bps: float
    ) -> Tuple[float, float]:
        """
        计算保留价格和半价差.

        使用Avellaneda-Stoikov模型的简化版本.
        """
        half_spread = mid * spread_bps / 10000 / 2

        # 库存倾斜
        if self.config.enable_inventory_skew:
            inventory_ratio = inventory / max(self.config.inventory_limit, 1.0)
            skew = inventory_ratio * half_spread * self.config.inventory_skew_factor
            reservation_price = mid - skew
        else:
            reservation_price = mid

        return reservation_price, half_spread

    def _calculate_quote_sizes(
        self, inventory: float, inventory_limit: float
    ) -> Tuple[float, float]:
        """计算报价数量."""
        base_size = self.config.quote_size

        # 根据库存调整
        inventory_ratio = inventory / max(inventory_limit, 1.0)

        # 买单减少当多头过多,卖单减少当空头过多
        bid_size = base_size * max(0, 1 - inventory_ratio)
        ask_size = base_size * max(0, 1 + inventory_ratio)

        return bid_size, ask_size

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """
        生成报价 (高性能版本).

        目标延迟: <35ms (P95)
        """
        import time

        start_time = time.perf_counter()

        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")

        circuit_state = self.circuit_breaker.state
        # 1. 检查熔断 (如果启用)
        if self.config.enable_circuit_breaker:
            if self._should_refresh_risk_check(state.timestamp, mid, position.size):
                portfolio = self._update_portfolio_state(state, position)
                circuit_state = self.circuit_breaker.check_risk_limits(portfolio)
                can_trade, reason = self.circuit_breaker.can_trade(TradeAction.MARKET_MAKING)
                self._last_can_trade = can_trade
                self._last_halt_reason = reason
                self._last_risk_check_at = state.timestamp
                self._last_risk_check_mid = mid
                self._last_risk_check_inventory = position.size
            else:
                can_trade = self._last_can_trade
                reason = self._last_halt_reason

            if not can_trade:
                return QuoteAction(
                    bid_price=mid,
                    bid_size=0.0,
                    ask_price=mid,
                    ask_size=0.0,
                    metadata={
                        "strategy": self.name,
                        "circuit_state": circuit_state.value,
                        "trading_halted": True,
                        "halt_reason": reason,
                    },
                )

        # 2. 更新状态检测 (Fast版本, <5ms)
        if self.config.enable_regime_detection:
            ret = self._calculate_return(mid)
            if ret is not None:
                self.regime_detector.update(ret)
                self._returns_history.append(ret)

        current_regime = self.regime_detector.current_regime

        # 3. 尝试从缓存获取Greeks (如果可用)
        # 使用价格区间作为缓存键，减少缓存抖动
        price_bucket = int(mid / 100) * 100 if mid > 0 else 0
        cache_key = f"{state.instrument}_{price_bucket}"
        # Prefer real-time greeks from current state, then cache fallback.
        if state.greeks is not None:
            self._set_cached_greeks(cache_key, state.greeks)
            self._current_greeks = state.greeks
        else:
            cached_greeks = self._get_cached_greeks(cache_key)
            # Ensure stale greeks are not reused when cache misses.
            self._current_greeks = cached_greeks

        # 4. 检查对冲
        hedge_decision = None
        if self.config.enable_adaptive_hedging and self._current_greeks:
            hedge_decision = self.hedger.should_hedge(
                state.timestamp, mid, self._current_greeks, position.size
            )

        # 4. 计算价差
        spread_multiplier = self._get_spread_multiplier()
        spread_bps = self.config.base_spread_bps * spread_multiplier

        # 5. 计算保留价格
        reservation_price, half_spread = self._calculate_reservation_price(
            mid, position.size, spread_bps
        )

        # 6. 应用仓位限制
        position_limit_mult = self.circuit_breaker.get_position_limit_multiplier()
        effective_inventory_limit = self.config.inventory_limit * position_limit_mult

        # 7. 确定报价数量
        bid_size, ask_size = self._calculate_quote_sizes(position.size, effective_inventory_limit)

        # 8. 构建元数据
        latency_ms = (time.perf_counter() - start_time) * 1000

        metadata = {
            "strategy": self.name,
            "circuit_state": (
                circuit_state.value if self.config.enable_circuit_breaker else "disabled"
            ),
            "regime": current_regime.name,
            "spread_bps": spread_bps,
            "spread_multiplier": spread_multiplier,
            "reservation_price": reservation_price,
            "half_spread": half_spread,
            "inventory": position.size,
            "latency_ms": latency_ms,
            "hedge_decision": hedge_decision.reason if hedge_decision else None,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

        # 更新统计
        self._quote_count += 1
        self._total_latency_ms += latency_ms

        return QuoteAction(
            bid_price=reservation_price - half_spread,
            bid_size=bid_size,
            ask_price=reservation_price + half_spread,
            ask_size=ask_size,
            metadata=metadata,
        )

    def get_performance_stats(self) -> Dict:
        """获取性能统计."""
        avg_latency = self._total_latency_ms / max(1, self._quote_count)
        cache_hit_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)

        return {
            "quote_count": self._quote_count,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "regime_stats": self.regime_detector.get_stats(),
        }

    def reset(self) -> None:
        """重置策略状态."""
        self._returns_history.clear()
        self._current_greeks = None
        self._greeks_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._quote_count = 0
        self._total_latency_ms = 0.0
        self._last_price = None
        self._last_risk_check_at = None
        self._last_risk_check_mid = None
        self._last_risk_check_inventory = None
        self._last_can_trade = True
        self._last_halt_reason = "ok"

        if hasattr(self.circuit_breaker, "reset"):
            self.circuit_breaker.reset()
        self.regime_detector.reset()
        if hasattr(self.hedger, "reset"):
            self.hedger.reset()

    def get_internal_state(self) -> Dict:
        """Return current internal state for debugging."""
        return {
            "strategy_name": self.name,
            "config": {
                "base_spread_bps": self.config.base_spread_bps,
                "inventory_limit": self.config.inventory_limit,
                "enable_circuit_breaker": self.config.enable_circuit_breaker,
                "enable_regime_detection": self.config.enable_regime_detection,
                "enable_adaptive_hedging": self.config.enable_adaptive_hedging,
            },
            "performance": self.get_performance_stats(),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "current_regime": self.regime_detector.current_regime.name,
            "cache_size": len(self._greeks_cache),
        }
