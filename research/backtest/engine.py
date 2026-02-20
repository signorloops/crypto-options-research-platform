"""
Event-driven backtest engine for market making strategies.
Simulates realistic market conditions including latency, fill probability, and adverse selection.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.types import (
    Fill,
    MarketState,
    OrderBook,
    OrderSide,
    Position,
    QuoteAction,
    Trade,
)
from research.pricing.inverse_options import InverseOptionPricer
from strategies.base import MarketMakingStrategy


@dataclass
class FillSimulatorConfig:
    """Configuration for fill simulation."""
    # Latency parameters (in milliseconds) - 从环境变量读取
    base_latency_ms: float = field(default_factory=lambda: float(os.getenv("BT_BASE_LATENCY_MS", "50.0")))
    latency_std_ms: float = field(default_factory=lambda: float(os.getenv("BT_LATENCY_STD_MS", "20.0")))

    # Queue position model
    queue_position_random: bool = field(default_factory=lambda: os.getenv("BT_QUEUE_POSITION_RANDOM", "true").lower() == "true")

    # Adverse selection - 从环境变量读取
    adverse_selection_factor: float = field(default_factory=lambda: float(os.getenv("BT_ADVERSE_SELECTION_FACTOR", "0.3")))

    # Minimum profitability (avoid fills that would be instant losses) - 从环境变量读取
    min_profit_bps: float = field(default_factory=lambda: float(os.getenv("BT_MIN_PROFIT_BPS", "0.5")))


class RealisticFillSimulator:
    """
    Simulates realistic order fills based on market microstructure.

    Key features:
    1. Queue position model: Orders at front of queue fill faster
    2. Latency simulation: Quote updates have delay
    3. Adverse selection: Informed trades hit stale quotes
    4. Size-based probability: Larger quotes less likely to fully fill
    """

    def __init__(self, config: FillSimulatorConfig = None, rng: Optional[np.random.Generator] = None):
        self.config = config or FillSimulatorConfig()
        self._quote_history: List[Dict] = []
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
        next_trades: List[Trade],
        inventory_pressure: float = 0.0,
        transaction_cost_bps: float = 0.0,
    ) -> Optional[Fill]:
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
    ) -> Optional[Fill]:
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
            fill_price *= (1 + cost_multiplier)
        else:
            fill_price *= (1 - cost_multiplier)
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
                fill_price *= (1 + adverse_slip)
            else:
                fill_price *= (1 - adverse_slip)
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
            quote_id=None
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def _queue_depth_ahead(self, quote: QuoteAction, side: OrderSide, order_book: OrderBook) -> float:
        """Approximate queue depth ahead of our quote."""
        if side == OrderSide.BUY:
            our_price = quote.bid_price
            levels = order_book.bids
            better = lambda p: p > our_price
            same = lambda p: p == our_price
        else:
            our_price = quote.ask_price
            levels = order_book.asks
            better = lambda p: p < our_price
            same = lambda p: p == our_price

        volume_ahead = 0.0
        for level in levels:
            if better(level.price):
                volume_ahead += level.size
            elif same(level.price):
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
        order_book: Optional[OrderBook],
        side: OrderSide
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
        size: float
    ) -> float:
        """Positive execution loss measured against reference price."""
        if side == OrderSide.BUY:
            return max(executed_price - reference_price, 0.0) * size
        return max(reference_price - executed_price, 0.0) * size


@dataclass
class BacktestResult:
    """Results from a backtest run (coin-margined)."""
    strategy_name: str

    # PnL metrics (in crypto units for coin-margined)
    total_pnl_crypto: float  # PnL in cryptocurrency units (BTC, ETH)
    total_pnl_usd: float  # PnL converted to USD at current price
    realized_pnl: float
    unrealized_pnl: float
    inventory_pnl: float  # PnL from inventory changes

    # Risk metrics
    sharpe_ratio: float
    deflated_sharpe_ratio: float
    max_drawdown: float
    volatility: float
    sharpe_ci_95: Tuple[float, float]
    drawdown_ci_95: Tuple[float, float]

    # Trade statistics
    trade_count: int
    buy_count: int
    sell_count: int
    avg_trade_size: float
    avg_trade_pnl_crypto: float

    # Spread capture
    total_spread_captured: float
    avg_spread_captured_bps: float

    # Costs
    inventory_cost: float  # Cost of holding inventory
    adverse_selection_cost: float  # Losses to informed traders

    # Crypto balance tracking (coin-margined specific)
    crypto_balance: float  # Balance in cryptocurrency units
    crypto_balance_series: pd.Series = field(default_factory=lambda: pd.Series())

    # Time series
    pnl_series: pd.Series = field(default_factory=lambda: pd.Series())
    inventory_series: pd.Series = field(default_factory=lambda: pd.Series())

    def summary(self) -> str:
        """Generate text summary of results."""
        return f"""
Backtest Results: {self.strategy_name} (Coin-Margined)
{'='*50}
Total PnL (Crypto): {self.total_pnl_crypto:.8f}
Total PnL (USD):    ${self.total_pnl_usd:,.2f}
Crypto Balance:     {self.crypto_balance:.8f}
Sharpe Ratio:       {self.sharpe_ratio:.2f}
Deflated Sharpe:    {self.deflated_sharpe_ratio:.2f}
Max Drawdown:       {self.max_drawdown:.2%}

Trade Statistics:
  Total Trades:     {self.trade_count}
  Buys:             {self.buy_count}
  Sells:            {self.sell_count}
  Avg Trade PnL:    {self.avg_trade_pnl_crypto:.8f} crypto

PnL Breakdown:
  Spread Capture:   ${self.total_spread_captured:,.2f}
  Inventory Cost:   ${self.inventory_cost:,.2f}
  Adverse Select:   ${self.adverse_selection_cost:,.2f}
"""


class BacktestEngine:
    """
    Event-driven backtest engine for market making strategies (coin-margined).

    Simulates:
    - Real-time quote updates
    - Fill probability based on queue position
    - Inventory tracking and PnL calculation (in crypto units)
    - Adverse selection effects
    - Coin-margined specific: PnL calculated in cryptocurrency units
    """

    def __init__(
        self,
        strategy: MarketMakingStrategy,
        fill_simulator: Optional[RealisticFillSimulator] = None,
        initial_crypto_balance: float = 1.0,
        base_currency: str = "BTC",
        random_seed: Optional[int] = None,
        transaction_cost_bps: float = 2.0,
    ):
        self.strategy = strategy
        self.rng = np.random.default_rng(random_seed)
        self.fill_simulator = fill_simulator or RealisticFillSimulator(rng=self.rng)
        self.fill_simulator.rng = self.rng
        self.initial_crypto_balance = initial_crypto_balance
        self.base_currency = base_currency
        self.random_seed = random_seed
        self.transaction_cost_bps = transaction_cost_bps

        # State tracking - coin-margined: track crypto balance directly
        self.crypto_balance = initial_crypto_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Fill] = []
        self.quotes: List[QuoteAction] = []

        # Metrics tracking (PnL in crypto units)
        self._pnl_history: List[tuple] = []
        self._inventory_history: List[tuple] = []
        self._crypto_balance_history: List[tuple] = []

        # Sampling configuration to limit memory usage
        self._history_sampling_interval: int = 10  # Record every Nth tick
        self._max_history_points: int = 1_000_000  # Limit to prevent memory exhaustion
        self._tick_counter: int = 0  # Counter for sampling

    def run(
        self,
        market_data: pd.DataFrame,
        price_column: str = "price",
        timestamp_column: str = "timestamp"
    ) -> BacktestResult:
        """
        Run backtest on historical market data.

        Args:
            market_data: DataFrame with market data
            price_column: Column name for price data
            timestamp_column: Column name for timestamps

        Returns:
            BacktestResult with performance metrics
        """
        # Reset state
        self.crypto_balance = self.initial_crypto_balance
        self.positions = {}
        self.trades = []
        self.quotes = []
        self._pnl_history = []
        self._inventory_history = []
        self._crypto_balance_history = []
        self._tick_counter = 0
        self.strategy.reset()

        # Reset isolated RNG for reproducible runs without touching global state.
        self.rng = np.random.default_rng(self.random_seed)
        if self.fill_simulator is not None:
            self.fill_simulator.rng = self.rng
            self.fill_simulator.reset_metrics()

        # Pre-extract arrays to avoid per-row dict allocation (PERF-6)
        prices = market_data[price_column].to_numpy(dtype=np.float64)
        timestamps_arr = market_data[timestamp_column].to_numpy()
        n_events = len(prices)

        if n_events == 0:
            return self._compute_result(current_price=0.0)

        # Pre-extract event volumes to avoid DataFrame -> list[dict] memory blow-up.
        event_volumes = self._prepare_event_volumes(market_data)

        current_ob = self._create_dummy_order_book(prices[0])
        previous_quote: Optional[QuoteAction] = None

        for i in range(n_events):
            price = float(prices[i])
            timestamp = timestamps_arr[i]

            current_ob = self._update_order_book(current_ob, price)

            market_state = MarketState(
                timestamp=timestamp,
                instrument="SYNTHETIC",
                spot_price=price,
                order_book=current_ob,
                recent_trades=[]
            )

            position = self.positions.get("SYNTHETIC", Position("SYNTHETIC", 0, 0))

            if previous_quote is not None and self.fill_simulator:
                synthetic_trades = self._generate_synthetic_trades(
                    market_state,
                    volume=float(event_volumes[i]),
                )
                if synthetic_trades:
                    fill = self.fill_simulator.simulate_fill(
                        previous_quote, market_state, synthetic_trades,
                        transaction_cost_bps=self.transaction_cost_bps,
                    )
                    if fill:
                        self._process_fill(fill, position, price)
                        self.strategy.on_fill(fill, self.positions.get("SYNTHETIC", position))
                        position = self.positions.get("SYNTHETIC", position)

            lagged_price = float(prices[i - 1]) if i > 0 else price
            lagged_market_state = MarketState(
                timestamp=market_state.timestamp,
                instrument=market_state.instrument,
                spot_price=lagged_price,
                order_book=market_state.order_book,
                recent_trades=market_state.recent_trades
            )
            new_quote = self.strategy.quote(lagged_market_state, position)
            self.quotes.append(new_quote)
            previous_quote = new_quote

            self._record_state(timestamp, price)

        return self._compute_result(current_price=float(prices[-1]))

    def _prepare_event_volumes(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare per-event activity volume without materializing row dictionaries."""
        if "volume" not in market_data.columns:
            return np.ones(len(market_data), dtype=np.float64)

        values = market_data["volume"].to_numpy(dtype=np.float64, copy=False)
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0)
        return np.maximum(values, 0.0)

    def _create_dummy_order_book(self, price: float) -> OrderBook:
        """Create a simple order book around given price."""
        from core.types import OrderBookLevel
        spread = price * 0.001  # 10 bps spread

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=price - spread/2, size=1.0)],
            asks=[OrderBookLevel(price=price + spread/2, size=1.0)]
        )

    def _update_order_book(self, ob: OrderBook, new_price: float) -> OrderBook:
        """Update order book with new price."""
        spread = ob.spread or new_price * 0.001
        from core.types import OrderBookLevel

        return OrderBook(
            timestamp=ob.timestamp,
            instrument=ob.instrument,
            bids=[OrderBookLevel(price=new_price - spread/2, size=1.0)],
            asks=[OrderBookLevel(price=new_price + spread/2, size=1.0)]
        )

    def _apply_slippage(
        self,
        quote_price: float,
        trade_size: float,
        order_book: OrderBook,
        side: OrderSide
    ) -> float:
        """Apply realistic slippage model based on order book depth.

        Args:
            quote_price: Original quote price
            trade_size: Size of the trade
            order_book: Current order book
            side: Side of the fill (BUY or SELL)

        Returns:
            Slipped price
        """
        # Get relevant order book side
        if side == OrderSide.BUY:
            levels = order_book.asks
        else:
            levels = order_book.bids

        if not levels:
            return quote_price

        # Calculate VWAP to fill the trade size
        remaining = trade_size
        total_cost = 0.0

        for level in levels:
            if remaining <= 0:
                break
            fill_at_level = min(remaining, level.size)
            total_cost += fill_at_level * level.price
            remaining -= fill_at_level

        # If we couldn't fill full size at available levels, add penalty
        if remaining > 0:
            # Use last level price plus penalty for remaining
            last_price = levels[-1].price
            total_cost += remaining * (last_price * 1.001)  # 10 bps penalty
            filled_size = trade_size
        else:
            filled_size = trade_size

        vwap = total_cost / filled_size if filled_size > 0 else quote_price

        # Add random slippage component (1 bps std)
        random_slippage = self.rng.normal(0, quote_price * 0.0001)

        return vwap + random_slippage

    def _calculate_fill_probability(
        self,
        quote: QuoteAction,
        order_book: OrderBook,
        side: OrderSide
    ) -> float:
        """Calculate probability of fill based on quote position in order book.

        Args:
            quote: Our quote
            order_book: Current order book
            side: Side of the quote

        Returns:
            Probability of fill (0-1)
        """
        if side == OrderSide.BUY:
            our_price = quote.bid_price
            our_size = quote.bid_size
            levels = order_book.bids
        else:
            our_price = quote.ask_price
            our_size = quote.ask_size
            levels = order_book.asks

        if not levels or our_size <= 0:
            return 0.0

        # Calculate queue position (volume ahead of us)
        volume_ahead = 0.0
        for level in levels:
            if side == OrderSide.BUY:
                if level.price > our_price:
                    volume_ahead += level.size
                elif level.price == our_price:
                    # Assume random queue position at same price
                    volume_ahead += level.size * 0.5
                    break
            else:  # SELL
                if level.price < our_price:
                    volume_ahead += level.size
                elif level.price == our_price:
                    volume_ahead += level.size * 0.5
                    break

        # Fill probability decreases with queue depth
        # Base probability for being at front of queue
        base_prob = 0.6

        # Queue penalty: longer queue = lower probability
        queue_penalty = min(0.4, volume_ahead / (volume_ahead + our_size))

        # Size penalty: larger orders less likely to fully fill
        size_penalty = min(0.2, our_size / 10.0)  # Assuming avg trade size ~1

        return max(0.05, base_prob - queue_penalty - size_penalty)

    def _simulate_fill_simple(
        self,
        quote: QuoteAction,
        current_price: float,
        price_change: float,
        order_book: OrderBook
    ) -> Optional[Fill]:
        """Enhanced fill simulation with slippage and realistic fill probability."""
        # Determine which side might get filled based on price movement
        if price_change > current_price * 0.0005:  # 5 bps move up
            side = OrderSide.SELL
            if quote.ask_size <= 0:
                return None
        elif price_change < -current_price * 0.0005:  # 5 bps move down
            side = OrderSide.BUY
            if quote.bid_size <= 0:
                return None
        else:
            # Small price move, check both sides with lower probability
            side = None

        if side is None:
            # Small moves may still fill with lower probability
            if abs(price_change) > current_price * 0.0002:  # 2 bps threshold
                # Random side with 10% chance
                if self.rng.random() < 0.1:
                    side = OrderSide.SELL if self.rng.random() < 0.5 else OrderSide.BUY
                else:
                    return None
            else:
                return None

        # Calculate fill probability
        fill_prob = self._calculate_fill_probability(quote, order_book, side)

        if self.rng.random() > fill_prob:
            return None

        # Determine fill size (may be partial for large orders)
        if side == OrderSide.SELL:
            base_size = min(quote.ask_size, 0.5)
        else:
            base_size = min(quote.bid_size, 0.5)

        # Partial fill probability
        if base_size > 0.3 and self.rng.random() < 0.3:
            fill_size = base_size * self.rng.uniform(0.3, 0.7)
        else:
            fill_size = base_size

        # Apply slippage
        quote_price = quote.ask_price if side == OrderSide.SELL else quote.bid_price
        fill_price = self._apply_slippage(quote_price, fill_size, order_book, side)

        return Fill(
            timestamp=datetime.now(timezone.utc),
            instrument="SYNTHETIC",
            side=side,
            price=fill_price,
            size=fill_size
        )

    def _generate_synthetic_trades(
        self,
        market_state: MarketState,
        volume: float = 1.0
    ) -> List[Trade]:
        """Generate synthetic trades based on current market activity (no look-ahead).

        Uses current-event volume only (no future leakage).
        """
        trades = []
        timestamp = market_state.timestamp

        # Generate 0-3 trades based on volume
        num_trades = min(3, max(0, int(volume * self.rng.random())))

        for _ in range(num_trades):
            # Random trade side (slight bias based on order book imbalance)
            imbalance = market_state.order_book.imbalance() if hasattr(market_state.order_book, 'imbalance') else 0
            side_bias = 0.5 + imbalance * 0.2  # Small bias towards buy if bids are heavier
            side = OrderSide.BUY if self.rng.random() < side_bias else OrderSide.SELL

            # Trade price around mid with some randomness
            mid = market_state.spot_price
            price_offset = self.rng.normal(0, mid * 0.0001)  # 1 bps std
            trade_price = mid + price_offset

            # Trade size proportional to volume
            trade_size = volume * self.rng.random() * 0.3

            trades.append(Trade(
                timestamp=timestamp,
                instrument=market_state.instrument,
                price=abs(trade_price),
                size=abs(trade_size),
                side=side
            ))

        return trades

    def _process_fill(self, fill: Fill, current_position: Position, current_price: float) -> None:
        """Process a fill and update portfolio (coin-margined).

        Note: This tracks premium cash flows for market making strategies.
        For complete PnL tracking including option position valuation,
        additional Greeks-based mark-to-market should be implemented.

        Args:
            fill: The executed fill
            current_position: Position before the fill
            current_price: Current underlying price for crypto conversion
        """
        self.trades.append(fill)

        # Update position
        new_position = current_position.apply_fill(fill)
        self.positions[fill.instrument] = new_position

        # Coin-margined premium calculation:
        # For coin-margined options, premium is paid/received in cryptocurrency
        # If fill.price is in USD: premium_crypto = (option_price_usd * size) / underlying_price_usd
        # If fill.price is already in crypto: premium_crypto = option_price_crypto * size
        # Assuming fill.price is in USD and size is in USD notional
        premium_crypto = (fill.price * fill.size) / current_price

        if fill.side == OrderSide.BUY:
            # Buying option: pay premium in crypto
            self.crypto_balance -= premium_crypto
        else:
            # Selling option: receive premium in crypto
            self.crypto_balance += premium_crypto

    def _record_state(self, timestamp: datetime, current_price: float) -> None:
        """Record current state for analysis with sampling to limit memory usage."""
        self._tick_counter += 1

        # Only record every Nth tick to reduce memory usage
        # But always record the first tick to ensure we have at least one data point
        if self._tick_counter % self._history_sampling_interval != 0 and self._tick_counter > 1:
            return

        position = self.positions.get("SYNTHETIC", Position("SYNTHETIC", 0, 0))
        self._inventory_history.append((timestamp, position.size))

        # Calculate PnL in crypto units (coin-margined)
        # For coin-margined: PnL = sum of (1/entry_price - 1/exit_price) * size for each trade
        crypto_pnl = self._calculate_crypto_pnl(current_price)
        self._pnl_history.append((timestamp, crypto_pnl))
        self._crypto_balance_history.append((timestamp, self.crypto_balance))

        # Limit history size to prevent memory exhaustion
        if len(self._pnl_history) > self._max_history_points:
            # Keep the most recent data, remove oldest 20%
            remove_count = self._max_history_points // 5
            self._pnl_history = self._pnl_history[remove_count:]
            self._inventory_history = self._inventory_history[remove_count:]
            self._crypto_balance_history = self._crypto_balance_history[remove_count:]

    def _calculate_crypto_pnl(self, current_price: Optional[float] = None) -> float:
        """Calculate PnL in cryptocurrency units (coin-margined formula).

        For coin-margined options:
        - Unrealized PnL = position_size * (1/entry_price - 1/current_price)
        - Plus realized PnL from premium cashflows (tracked in crypto_balance)

        Args:
            current_price: Current market price. If None, only realized PnL is returned.
        """
        # Start with realized PnL from cash balance
        realized_pnl = self.crypto_balance - self.initial_crypto_balance

        # Add unrealized PnL from open positions (if current price available)
        unrealized_pnl = 0.0
        if current_price is not None and current_price > 0:
            for instrument, position in self.positions.items():
                if position.size != 0 and position.avg_entry_price > 0:
                    # Use inverse option PnL formula
                    unrealized_pnl += InverseOptionPricer.calculate_pnl(
                        entry_price=position.avg_entry_price,
                        exit_price=current_price,
                        size=position.size,
                        inverse=True
                    )

        return realized_pnl + unrealized_pnl

    def _bootstrap_risk_ci(
        self,
        returns: pd.Series,
        n_bootstrap: int = 500,
        alpha: float = 0.05
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Bootstrap 95% CI for Sharpe and max drawdown.
        """
        if len(returns) < 10:
            return (0.0, 0.0), (0.0, 0.0)

        sharpe_samples: List[float] = []
        dd_samples: List[float] = []
        annualization = np.sqrt(365.0)
        values = returns.to_numpy()

        for _ in range(n_bootstrap):
            sample = self.rng.choice(values, size=len(values), replace=True)
            std = np.std(sample, ddof=1)
            if std > 1e-12:
                sharpe_samples.append(float(np.mean(sample) / std * annualization))
            else:
                sharpe_samples.append(0.0)

            cum = np.cumsum(sample)
            running_max = np.maximum.accumulate(cum)
            denom = np.where(np.abs(running_max) > 1e-12, np.abs(running_max), 1.0)
            dd = (running_max - cum) / denom
            dd_samples.append(float(np.max(dd)))

        lo = alpha / 2
        hi = 1 - alpha / 2
        sharpe_ci = (
            float(np.quantile(sharpe_samples, lo)),
            float(np.quantile(sharpe_samples, hi)),
        )
        drawdown_ci = (
            float(np.quantile(dd_samples, lo)),
            float(np.quantile(dd_samples, hi)),
        )
        return sharpe_ci, drawdown_ci

    @staticmethod
    def _deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int = 1) -> float:
        """
        Simplified Deflated Sharpe Ratio approximation.
        """
        if n_obs < 5:
            return 0.0
        # Expected max SR from multiple testing under Gaussian null.
        expected_max_sr = norm.ppf(1.0 - 1.0 / max(n_trials, 2)) / np.sqrt(max(n_obs - 1, 1))
        denom = max(1e-12, np.sqrt(1.0 / max(n_obs - 1, 1)))
        z = (sharpe - expected_max_sr) / denom
        return float(norm.cdf(z))

    def _compute_result(self, current_price: Optional[float] = None) -> BacktestResult:
        """Compute final backtest metrics (coin-margined)."""
        # Convert histories to series
        if self._pnl_history:
            pnl_series = pd.Series(
                [x[1] for x in self._pnl_history],
                index=[x[0] for x in self._pnl_history]
            )
        else:
            pnl_series = pd.Series()

        if self._inventory_history:
            inventory_series = pd.Series(
                [x[1] for x in self._inventory_history],
                index=[x[0] for x in self._inventory_history]
            )
        else:
            inventory_series = pd.Series()

        if self._crypto_balance_history:
            crypto_balance_series = pd.Series(
                [x[1] for x in self._crypto_balance_history],
                index=[x[0] for x in self._crypto_balance_history]
            )
        else:
            crypto_balance_series = pd.Series()

        # Calculate metrics (PnL in crypto units)
        total_pnl_crypto = float(pnl_series.iloc[-1]) if len(pnl_series) > 0 else 0.0
        total_pnl_usd = total_pnl_crypto * current_price  # Convert to USD for reference

        # Sharpe ratio calculation using crypto PnL returns
        if len(pnl_series) > 1:
            # Calculate returns based on PnL changes (not balance)
            # Use log returns for better statistical properties
            pnl_changes = pnl_series.diff().dropna()
            returns = pnl_changes / self.initial_crypto_balance  # Normalized by initial capital

            if returns.std() > 0:
                # Calculate annualization factor based on data frequency
                if len(returns) > 1:
                    # Estimate periods per year from data timestamps
                    time_span = (pnl_series.index[-1] - pnl_series.index[0]).total_seconds()
                    periods_per_year = len(returns) * (365.0 * 24 * 3600) / max(time_span, 1)
                    annualization = np.sqrt(periods_per_year)
                else:
                    annualization = np.sqrt(365.0 * 24)  # Default hourly

                # Risk-free rate (crypto can use 0 or small positive rate)
                risk_free_rate = 0.0
                excess_returns = returns - risk_free_rate / (365.0 * 24)  # Convert to period rate
                sharpe = (excess_returns.mean() / returns.std()) * annualization
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown calculation
        # Correct formula: Drawdown = (Peak - Current) / Peak
        if len(pnl_series) > 0:
            running_max = pnl_series.expanding().max()
            # Avoid division by zero
            running_max_safe = running_max.replace(0, np.nan)
            drawdown = (running_max_safe - pnl_series) / running_max_safe
            max_dd = drawdown.max()
            if np.isnan(max_dd):
                max_dd = 0.0
        else:
            max_dd = 0.0

        # Trade statistics
        buys = len([t for t in self.trades if t.side == OrderSide.BUY])
        sells = len([t for t in self.trades if t.side == OrderSide.SELL])
        avg_size = np.mean([t.size for t in self.trades]) if self.trades else 0
        returns_for_ci = pnl_series.diff().dropna() / max(self.initial_crypto_balance, 1e-12) if len(pnl_series) > 1 else pd.Series(dtype=float)
        sharpe_ci, drawdown_ci = self._bootstrap_risk_ci(returns_for_ci)
        deflated_sharpe = self._deflated_sharpe_ratio(float(sharpe), n_obs=len(returns_for_ci), n_trials=1)
        execution_cost = 0.0
        adverse_selection_cost = 0.0
        if self.fill_simulator is not None:
            execution_cost = float(
                self.fill_simulator.transaction_cost_paid + self.fill_simulator.slippage_cost
            )
            adverse_selection_cost = float(self.fill_simulator.adverse_selection_cost)

        return BacktestResult(
            strategy_name=self.strategy.name,
            total_pnl_crypto=total_pnl_crypto,
            total_pnl_usd=total_pnl_usd,
            realized_pnl=total_pnl_crypto,  # Simplified
            unrealized_pnl=0,
            inventory_pnl=0,
            sharpe_ratio=sharpe,
            deflated_sharpe_ratio=deflated_sharpe,
            max_drawdown=max_dd,
            volatility=pnl_series.std() if len(pnl_series) > 1 else 0,
            sharpe_ci_95=sharpe_ci,
            drawdown_ci_95=drawdown_ci,
            trade_count=len(self.trades),
            buy_count=buys,
            sell_count=sells,
            avg_trade_size=avg_size,
            avg_trade_pnl_crypto=total_pnl_crypto / len(self.trades) if self.trades else 0,
            total_spread_captured=0,
            avg_spread_captured_bps=0,
            inventory_cost=execution_cost,
            adverse_selection_cost=adverse_selection_cost,
            crypto_balance=self.crypto_balance,
            crypto_balance_series=crypto_balance_series,
            pnl_series=pnl_series,
            inventory_series=inventory_series
        )
