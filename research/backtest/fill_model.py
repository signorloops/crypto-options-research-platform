"""Fill simulation models for backtest execution realism."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from core.types import Fill, MarketState, OrderBook, OrderSide, QuoteAction, Trade


@dataclass
class FillSimulatorConfig:
    """Configuration for fill simulation."""

    # Latency parameters (in milliseconds) - 从环境变量读取
    base_latency_ms: float = field(
        default_factory=lambda: float(os.getenv("BT_BASE_LATENCY_MS", "50.0"))
    )
    latency_std_ms: float = field(
        default_factory=lambda: float(os.getenv("BT_LATENCY_STD_MS", "20.0"))
    )

    # Queue position model
    queue_position_random: bool = field(
        default_factory=lambda: os.getenv("BT_QUEUE_POSITION_RANDOM", "true").lower() == "true"
    )

    # Adverse selection - 从环境变量读取
    adverse_selection_factor: float = field(
        default_factory=lambda: float(os.getenv("BT_ADVERSE_SELECTION_FACTOR", "0.3"))
    )

    # Minimum profitability (avoid fills that would be instant losses) - 从环境变量读取
    min_profit_bps: float = field(
        default_factory=lambda: float(os.getenv("BT_MIN_PROFIT_BPS", "0.5"))
    )


def _apply_transaction_cost_to_fill(
    *,
    fill_price: float,
    side: OrderSide,
    size: float,
    transaction_cost_bps: float,
    cost_against_side_fn,
) -> tuple[float, float]:
    cost_multiplier = transaction_cost_bps / 10_000
    pre_fee_price = fill_price
    if side == OrderSide.BUY:
        fill_price *= 1 + cost_multiplier
    else:
        fill_price *= 1 - cost_multiplier
    cost = cost_against_side_fn(
        reference_price=pre_fee_price,
        executed_price=fill_price,
        side=side,
        size=size,
    )
    return fill_price, float(cost)


def _apply_adverse_selection_slippage(
    *,
    fill_price: float,
    side: OrderSide,
    size: float,
    adverse_selection_factor: float,
    is_adverse: bool,
    cost_against_side_fn,
) -> tuple[float, float]:
    if not is_adverse:
        return fill_price, 0.0
    adverse_slip = adverse_selection_factor * 0.001
    pre_adverse_price = fill_price
    if side == OrderSide.BUY:
        fill_price *= 1 + adverse_slip
    else:
        fill_price *= 1 - adverse_slip
    cost = cost_against_side_fn(
        reference_price=pre_adverse_price,
        executed_price=fill_price,
        side=side,
        size=size,
    )
    return fill_price, float(cost)


def _apply_slippage_to_fill(
    *,
    base_price: float,
    trade_size: float,
    order_book: OrderBook,
    side: OrderSide,
    apply_order_book_slippage_fn,
    cost_against_side_fn,
) -> tuple[float, float]:
    fill_price = apply_order_book_slippage_fn(
        quote_price=base_price,
        trade_size=trade_size,
        order_book=order_book,
        side=side,
    )
    slippage_cost = cost_against_side_fn(
        reference_price=base_price,
        executed_price=fill_price,
        side=side,
        size=trade_size,
    )
    return fill_price, float(slippage_cost)


def _apply_post_slippage_costs(
    *,
    fill_price: float,
    side: OrderSide,
    size: float,
    transaction_cost_bps: float,
    adverse_selection_factor: float,
    is_adverse: bool,
    cost_against_side_fn,
) -> tuple[float, float, float]:
    fill_price, transaction_cost = _apply_transaction_cost_to_fill(
        fill_price=fill_price,
        side=side,
        size=size,
        transaction_cost_bps=transaction_cost_bps,
        cost_against_side_fn=cost_against_side_fn,
    )
    fill_price, adverse_selection_cost = _apply_adverse_selection_slippage(
        fill_price=fill_price,
        side=side,
        size=size,
        adverse_selection_factor=adverse_selection_factor,
        is_adverse=is_adverse,
        cost_against_side_fn=cost_against_side_fn,
    )
    return fill_price, float(transaction_cost), float(adverse_selection_cost)


class RealisticFillSimulator:
    """Simulate realistic order fills based on market microstructure."""

    def __init__(
        self,
        config: FillSimulatorConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config or FillSimulatorConfig()
        self._quote_history: list[dict] = []
        self.rng = rng or np.random.default_rng()
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """Reset cumulative execution friction metrics."""
        self.transaction_cost_paid: float = 0.0
        self.slippage_cost: float = 0.0
        self.adverse_selection_cost: float = 0.0

    def simulate_fill(
        self,
        quote: QuoteAction,
        market_state: MarketState,
        next_trades: list[Trade],
        quote_timestamp=None,
        inventory_pressure: float = 0.0,
        transaction_cost_bps: float = 0.0,
    ) -> Fill | None:
        """Simulate whether a quote gets filled by incoming trades."""
        if not next_trades:
            return None
        if quote_timestamp is None:
            quote_timestamp = market_state.timestamp
        if self.config.latency_std_ms <= 0:
            latency_ms = max(0.0, self.config.base_latency_ms)
        else:
            latency_ms = max(
                0.0,
                self.rng.normal(self.config.base_latency_ms, self.config.latency_std_ms),
            )
        for trade in next_trades:
            trade_delay_ms = self._time_diff_ms(trade.timestamp, quote_timestamp)
            if trade_delay_ms < latency_ms:
                continue
            if trade.side == OrderSide.SELL:
                if trade.price <= quote.bid_price and quote.bid_size > 0:
                    return self._create_fill(trade, quote, OrderSide.BUY, market_state, transaction_cost_bps, latency_ms=latency_ms, inventory_pressure=inventory_pressure)
            elif trade.price >= quote.ask_price and quote.ask_size > 0:
                return self._create_fill(trade, quote, OrderSide.SELL, market_state, transaction_cost_bps, latency_ms=latency_ms, inventory_pressure=inventory_pressure)
        return None

    def _create_fill(
        self,
        trade: Trade,
        quote: QuoteAction,
        our_side: OrderSide,
        market_state: MarketState,
        transaction_cost_bps: float = 0.0,
        latency_ms: float = 0.0,
        inventory_pressure: float = 0.0,
    ) -> Fill | None:
        """Create a fill object with realistic sizing, slippage, and costs."""
        our_size = quote.bid_size if our_side == OrderSide.BUY else quote.ask_size
        if our_size <= 0:
            return None
        fill_prob = self._estimate_fill_probability(quote=quote, trade=trade, our_side=our_side, market_state=market_state, latency_ms=latency_ms, inventory_pressure=inventory_pressure)
        if self.rng.random() > fill_prob:
            return None
        fill_size = min(trade.size, our_size)
        base_price = quote.bid_price if our_side == OrderSide.BUY else quote.ask_price
        fill_price, slippage_cost = _apply_slippage_to_fill(base_price=base_price, trade_size=fill_size, order_book=market_state.order_book, side=our_side, apply_order_book_slippage_fn=self._apply_order_book_slippage, cost_against_side_fn=self._cost_against_side)
        self.slippage_cost += slippage_cost
        fill_price, transaction_cost, adverse_selection_cost = _apply_post_slippage_costs(fill_price=fill_price, side=our_side, size=fill_size, transaction_cost_bps=transaction_cost_bps, adverse_selection_factor=self.config.adverse_selection_factor, is_adverse=self._check_adverse_selection(trade, market_state), cost_against_side_fn=self._cost_against_side)
        self.transaction_cost_paid += transaction_cost
        self.adverse_selection_cost += adverse_selection_cost
        return Fill(
            timestamp=trade.timestamp,
            instrument=market_state.instrument,
            side=our_side,
            price=fill_price,
            size=fill_size,
            quote_id=None,
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def _queue_depth_ahead(
        self, quote: QuoteAction, side: OrderSide, order_book: OrderBook
    ) -> float:
        """Approximate queue depth ahead of our quote."""
        if side == OrderSide.BUY:
            our_price = quote.bid_price
            levels = order_book.bids
        else:
            our_price = quote.ask_price
            levels = order_book.asks

        volume_ahead = 0.0
        for level in levels:
            if side == OrderSide.BUY and level.price > our_price:
                volume_ahead += level.size
            elif side == OrderSide.SELL and level.price < our_price:
                volume_ahead += level.size
            elif level.price == our_price:
                # Random queue placement approximation: half of same-price depth ahead.
                volume_ahead += 0.5 * level.size
                break
            else:
                break
        return float(max(volume_ahead, 0.0))

    @staticmethod
    def _short_horizon_volatility(market_state: MarketState) -> float:
        """Estimate short-horizon realized volatility from recent trade prices."""
        trades = market_state.recent_trades or []
        if len(trades) < 3:
            return 0.0
        prices = np.array([max(float(t.price), 1e-12) for t in trades], dtype=float)
        returns = np.diff(np.log(prices))
        if len(returns) == 0:
            return 0.0
        return float(np.std(returns))

    def _estimate_fill_probability(
        self,
        quote: QuoteAction,
        trade: Trade,
        our_side: OrderSide,
        market_state: MarketState,
        latency_ms: float = 0.0,
        inventory_pressure: float = 0.0,
    ) -> float:
        """Estimate fill probability from microstructure features."""
        order_book = market_state.order_book
        if our_side == OrderSide.BUY:
            our_price, our_size = quote.bid_price, max(quote.bid_size, 1e-8)
            competitiveness = max(0.0, our_price - trade.price) / max(abs(our_price), 1e-8)
            imbalance_term = float(order_book.imbalance(levels=5))
        else:
            our_price, our_size = quote.ask_price, max(quote.ask_size, 1e-8)
            competitiveness = max(0.0, trade.price - our_price) / max(abs(our_price), 1e-8)
            imbalance_term = float(-order_book.imbalance(levels=5))
        queue_ratio = (queue_ahead := self._queue_depth_ahead(quote, our_side, order_book)) / our_size; size_ratio = max(float(trade.size), 0.0) / our_size
        vol = self._short_horizon_volatility(market_state); latency_scale = max(self.config.base_latency_ms + self.config.latency_std_ms + 1.0, 1.0)
        latency_penalty = max(latency_ms, 0.0) / latency_scale
        score = (
            1.8
            + 45.0 * competitiveness
            - 0.7 * queue_ratio
            - 0.45 * size_ratio
            - 15.0 * vol
            - 0.3 * latency_penalty
            - 0.25 * max(imbalance_term, 0.0)
            - 0.3 * abs(inventory_pressure)
        )
        prob = self._sigmoid(score); return float(np.clip(prob, 0.02, 0.98))

    def _check_adverse_selection(self, trade: Trade, market_state: MarketState) -> bool:
        """Check if this trade represents informed flow."""
        # Simple heuristic: large trades more likely to be informed
        avg_trade_size = 0.1  # Assume average
        if trade.size > 3 * avg_trade_size:
            return self.rng.random() < self.config.adverse_selection_factor * 2
        return self.rng.random() < self.config.adverse_selection_factor

    @staticmethod
    def _time_diff_ms(later_timestamp, earlier_timestamp) -> float:
        """Return later-earlier in milliseconds across datetime/timestamp types."""
        delta = later_timestamp - earlier_timestamp
        if hasattr(delta, "total_seconds"):
            return float(delta.total_seconds() * 1000)
        return float(delta / np.timedelta64(1, "ms"))

    def _apply_order_book_slippage(
        self,
        quote_price: float,
        trade_size: float,
        order_book: OrderBook | None,
        side: OrderSide,
    ) -> float:
        """Apply size/depth-aware slippage around quoted price."""
        if order_book is None:
            return quote_price

        # Maker fills should be modeled on our quoted side of the book.
        levels = order_book.bids if side == OrderSide.BUY else order_book.asks
        if not levels or trade_size <= 0:
            return quote_price
        remaining = trade_size; notional = 0.0
        for level in levels:
            if remaining <= 0: break
            take = min(remaining, level.size)
            notional += take * level.price
            remaining -= take
        if remaining > 0:
            penalty = quote_price * (1.0005 if side == OrderSide.BUY else 0.9995)
            notional += remaining * penalty
        vwap = notional / trade_size
        random_slip = abs(float(self.rng.normal(0.0, quote_price * 0.0001)))
        if side == OrderSide.BUY:
            # No price-improvement assumption for passive fills.
            return float(max(vwap, quote_price) + random_slip)
        return float(max(min(vwap, quote_price) - random_slip, 0.0))

    @staticmethod
    def _cost_against_side(
        reference_price: float,
        executed_price: float,
        side: OrderSide,
        size: float,
    ) -> float:
        """Positive execution loss measured against reference price."""
        if side == OrderSide.BUY:
            return max(executed_price - reference_price, 0.0) * size
        return max(reference_price - executed_price, 0.0) * size
