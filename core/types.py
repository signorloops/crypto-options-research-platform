"""
Core type definitions for the crypto options research platform.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np


class OptionType(Enum):
    CALL = "C"
    PUT = "P"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Tick:
    """Market tick data."""
    timestamp: datetime
    instrument: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    mid: float = field(init=False)
    spread: float = field(init=False)
    spread_bps: float = field(init=False)

    def __post_init__(self) -> None:
        self.mid = (self.bid + self.ask) / 2
        self.spread = self.ask - self.bid
        self.spread_bps = (self.spread / self.mid) * 10000 if self.mid > 0 else 0


@dataclass
class Trade:
    """Trade execution data."""
    timestamp: datetime
    instrument: str
    price: float
    size: float
    side: OrderSide
    trade_id: Optional[str] = None


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: float
    num_orders: Optional[int] = None


@dataclass
class OrderBook:
    """Order book snapshot."""
    timestamp: datetime
    instrument: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        return (bb + ba) / 2 if bb is not None and ba is not None else None

    @property
    def spread(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        return ba - bb if bb and ba else None

    def imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance."""
        bid_vol = sum(lvl.size for lvl in self.bids[:levels])
        ask_vol = sum(lvl.size for lvl in self.asks[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0


@dataclass
class Greeks:
    """Option Greeks."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: Optional[float] = None
    vanna: Optional[float] = None
    charm: Optional[float] = None


@dataclass
class OptionContract:
    """Option contract specification (coin-margined only)."""
    underlying: str  # BTC-USD, ETH-USD (coin-margined)
    strike: float
    expiry: datetime
    option_type: OptionType

    # Exchange and symbol info (for multi-exchange support)
    exchange: Optional[str] = None
    symbol: Optional[str] = None  # Exchange-specific symbol

    # Coin-margined (inverse) flag - always True for this platform
    inverse: bool = True

    # Contract specs
    lot_size: float = 1.0
    tick_size: float = 0.01

    @property
    def instrument_name(self) -> str:
        """Generate standard instrument name."""
        expiry_str = self.expiry.strftime("%d%b%y").upper()
        opt_type = self.option_type.value
        return f"{self.underlying}-{expiry_str}-{int(self.strike)}-{opt_type}"

    @property
    def is_coin_margined(self) -> bool:
        """Check if this is a coin-margined (inverse) option."""
        return self.inverse

    @property
    def base_currency(self) -> str:
        """Get base currency (e.g., BTC from BTC-USD).

        Returns:
            The base currency symbol (e.g., "BTC", "ETH")
        """
        return self.underlying.split("-")[0] if "-" in self.underlying else self.underlying

    @property
    def quote_currency(self) -> str:
        """Get quote currency (always USD for coin-margined).

        Returns:
            "USD" for coin-margined options
        """
        return "USD"

    def time_to_expiry(self, as_of: datetime) -> float:
        """Time to expiry in years."""
        return max(0, (self.expiry - as_of).total_seconds() / (365.25 * 24 * 3600))


@dataclass
class MarketState:
    """Complete market state for strategy decisions."""
    timestamp: datetime
    instrument: str
    spot_price: float
    order_book: OrderBook
    recent_trades: List[Trade]

    # Optional fields for options
    implied_vol: Optional[float] = None
    greeks: Optional[Greeks] = None

    # Pre-computed features
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class Quote:
    """Market making quote."""
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2


@dataclass
class QuoteAction:
    """Strategy output with metadata."""
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    metadata: Dict = field(default_factory=dict)

    def to_quote(self) -> Quote:
        return Quote(
            bid_price=self.bid_price,
            bid_size=self.bid_size,
            ask_price=self.ask_price,
            ask_size=self.ask_size
        )


@dataclass
class Fill:
    """Execution fill."""
    timestamp: datetime
    instrument: str
    side: OrderSide
    price: float
    size: float
    quote_id: Optional[str] = None


@dataclass
class Position:
    """Current position."""
    instrument: str
    size: float  # Positive = long, negative = short
    avg_entry_price: float

    def unrealized_pnl(self, mark_price: float, inverse: bool = False) -> float:
        """Calculate unrealized PnL.

        Args:
            mark_price: Current market price
            inverse: If True, use coin-margined (inverse) formula:
                    PnL = size * (1/entry - 1/mark)
                    If False, use linear formula:
                    PnL = size * (mark - entry)

        Returns:
            Unrealized PnL in quote currency (USD)
        """
        if inverse:
            # Coin-margined (inverse) formula
            # For coin-margined options, PnL is in cryptocurrency
            # Convert to USD equivalent for reporting
            if self.avg_entry_price == 0 or mark_price == 0:
                return 0.0
            return self.size * (1.0 / self.avg_entry_price - 1.0 / mark_price)
        else:
            # Linear formula (U-margined)
            return self.size * (mark_price - self.avg_entry_price)

    def apply_fill(self, fill: Fill) -> "Position":
        """Update position with new fill."""
        if self.instrument != fill.instrument:
            raise ValueError("Instrument mismatch")

        signed_fill = fill.size if fill.side == OrderSide.BUY else -fill.size
        new_size = self.size + signed_fill

        # Same-direction trade: volume-weight average entry.
        if self.size == 0 or np.sign(self.size) == np.sign(signed_fill):
            if np.isclose(new_size, 0.0):
                new_avg = 0.0
            else:
                cost = self.size * self.avg_entry_price + signed_fill * fill.price
                new_avg = cost / new_size
        # Reduced without flipping: keep historical average for the remaining lot.
        elif np.sign(self.size) == np.sign(new_size) or np.isclose(new_size, 0.0):
            new_avg = 0.0 if np.isclose(new_size, 0.0) else self.avg_entry_price
        # Flipped direction: remaining position is newly opened at fill price.
        else:
            new_avg = fill.price

        return Position(
            instrument=self.instrument,
            size=new_size,
            avg_entry_price=new_avg
        )


@dataclass
class Portfolio:
    """Portfolio state."""
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0

    def get_position(self, instrument: str) -> Position:
        return self.positions.get(instrument, Position(instrument, 0, 0))

    def apply_fill(self, fill: Fill) -> None:
        """Apply fill to portfolio."""
        pos = self.get_position(fill.instrument)
        self.positions[fill.instrument] = pos.apply_fill(fill)

        # Update cash
        notional = fill.price * fill.size
        if fill.side == OrderSide.BUY:
            self.cash -= notional
        else:
            self.cash += notional

    def total_notional(self, mark_prices: Dict[str, float]) -> float:
        """Total notional exposure."""
        return sum(
            abs(pos.size) * mark_prices.get(inst, 0)
            for inst, pos in self.positions.items()
        )


# Type aliases
FeatureExtractor = Callable[[MarketState], Dict[str, float]]
