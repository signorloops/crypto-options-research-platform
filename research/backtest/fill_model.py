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


class RealisticFillSimulator:
    """
    Simulates realistic order fills based on market microstructure.

    Key features:
    1. Queue position model: Orders at front of queue fill faster
    2. Latency simulation: Quote updates have delay
    3. Adverse selection: Informed trades hit stale quotes
    4. Size-based probability: Larger quotes less likely to fully fill
    """

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
        inventory_pressure: float = 0.0,
        transaction_cost_bps: float = 0.0,
    ) -> Fill | None:
        """
        Simulate whether a quote gets filled by incoming trades.

        Args:
            quote: The quote we placed
            market_state: Current market state
            next_trades: Trades that occur after our quote
            inventory_pressure: How much we're pushing inventory limits

        Returns:
            Fill object if filled, None otherwise
        """
        if not next_trades:
            return None

        # Simulate latency: our quote arrives after some delay
        if self.config.latency_std_ms <= 0:
            latency_ms = max(0.0, self.config.base_latency_ms)
        else:
            latency_ms = max(
                0.0,
                self.rng.normal(self.config.base_latency_ms, self.config.latency_std_ms),
            )

        # Check each trade against our quote
        for trade in next_trades:
            # Check if trade timestamp is after our quote + latency
            trade_delay_ms = (trade.timestamp - market_state.timestamp).total_seconds() * 1000
            if trade_delay_ms < latency_ms:
                continue  # Trade happened before our quote arrived

            # Check if trade hits our quote
            if trade.side == OrderSide.SELL:
                # Seller hits bids - check if our bid is competitive
                if trade.price <= quote.bid_price and quote.bid_size > 0:
                    return self._create_fill(
                        trade,
                        quote,
                        OrderSide.BUY,
                        market_state,
                        transaction_cost_bps,
                        latency_ms=latency_ms,
                        inventory_pressure=inventory_pressure,
                    )
            else:
                # Buyer lifts asks - check if our ask is competitive
                if trade.price >= quote.ask_price and quote.ask_size > 0:
                    return self._create_fill(
                        trade,
                        quote,
                        OrderSide.SELL,
                        market_state,
                        transaction_cost_bps,
                        latency_ms=latency_ms,
                        inventory_pressure=inventory_pressure,
                    )

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

        fill_prob = self._estimate_fill_probability(
            quote=quote,
            trade=trade,
            our_side=our_side,
            market_state=market_state,
            latency_ms=latency_ms,
            inventory_pressure=inventory_pressure,
        )
        if self.rng.random() > fill_prob:
            return None

        fill_size = min(trade.size, our_size)

        base_price = quote.bid_price if our_side == OrderSide.BUY else quote.ask_price
        fill_price = self._apply_order_book_slippage(
            quote_price=base_price,
            trade_size=fill_size,
            order_book=market_state.order_book,
            side=our_side,
        )
        self.slippage_cost += self._cost_against_side(
            reference_price=base_price,
            executed_price=fill_price,
            side=our_side,
            size=fill_size,
        )

        # Apply transaction cost: buyer pays more, seller receives less
        cost_multiplier = transaction_cost_bps / 10_000
        pre_fee_price = fill_price
        if our_side == OrderSide.BUY:
            fill_price *= 1 + cost_multiplier
        else:
            fill_price *= 1 - cost_multiplier
        self.transaction_cost_paid += self._cost_against_side(
            reference_price=pre_fee_price,
            executed_price=fill_price,
            side=our_side,
            size=fill_size,
        )

        # Adverse selection: additional slippage against us
        is_adverse = self._check_adverse_selection(trade, market_state)
        if is_adverse:
            adverse_slip = self.config.adverse_selection_factor * 0.001
            pre_adverse_price = fill_price
            if our_side == OrderSide.BUY:
                fill_price *= 1 + adverse_slip
            else:
                fill_price *= 1 - adverse_slip
            self.adverse_selection_cost += self._cost_against_side(
                reference_price=pre_adverse_price,
                executed_price=fill_price,
                side=our_side,
                size=fill_size,
            )

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
            our_price = quote.bid_price
            our_size = max(quote.bid_size, 1e-8)
            competitiveness = max(0.0, our_price - trade.price) / max(abs(our_price), 1e-8)
            imbalance_term = float(order_book.imbalance(levels=5))
        else:
            our_price = quote.ask_price
            our_size = max(quote.ask_size, 1e-8)
            competitiveness = max(0.0, trade.price - our_price) / max(abs(our_price), 1e-8)
            imbalance_term = float(-order_book.imbalance(levels=5))

        queue_ahead = self._queue_depth_ahead(quote, our_side, order_book)
        queue_ratio = queue_ahead / our_size
        size_ratio = max(float(trade.size), 0.0) / our_size
        vol = self._short_horizon_volatility(market_state)

        latency_scale = max(self.config.base_latency_ms + self.config.latency_std_ms + 1.0, 1.0)
        latency_penalty = max(latency_ms, 0.0) / latency_scale

        # Logistic score calibrated to preserve old baseline behavior while using features.
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
        prob = self._sigmoid(score)
        return float(np.clip(prob, 0.02, 0.98))

    def _check_adverse_selection(self, trade: Trade, market_state: MarketState) -> bool:
        """Check if this trade represents informed flow."""
        # Simple heuristic: large trades more likely to be informed
        avg_trade_size = 0.1  # Assume average
        if trade.size > 3 * avg_trade_size:
            return self.rng.random() < self.config.adverse_selection_factor * 2
        return self.rng.random() < self.config.adverse_selection_factor

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

        levels = order_book.asks if side == OrderSide.BUY else order_book.bids
        if not levels or trade_size <= 0:
            return quote_price

        remaining = trade_size
        notional = 0.0

        for level in levels:
            if remaining <= 0:
                break
            take = min(remaining, level.size)
            notional += take * level.price
            remaining -= take

        if remaining > 0:
            worst_price = levels[-1].price
            penalty = worst_price * (1.001 if side == OrderSide.BUY else 0.999)
            notional += remaining * penalty

        vwap = notional / trade_size
        random_slip = self.rng.normal(0.0, quote_price * 0.0001)
        return float(vwap + random_slip)

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
