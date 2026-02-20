"""
Pydantic schemas for data validation.

Compatible with Pydantic v2 (Python 3.8+).
"""
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

if TYPE_CHECKING:
    from core.types import OrderBook, Tick, Trade


class OptionType(str, Enum):
    """Option type enumeration."""
    CALL = "CALL"
    PUT = "PUT"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class MarketType(str, Enum):
    """Market type enumeration."""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"


class TickData(BaseModel):
    """Validated tick data schema."""
    timestamp: datetime
    instrument: str = Field(..., min_length=1, max_length=50)
    bid: float = Field(..., ge=0)
    ask: float = Field(..., ge=0)
    bid_size: float = Field(..., ge=0)
    ask_size: float = Field(..., ge=0)

    @field_validator('ask')
    @classmethod
    def ask_greater_than_bid(cls, v: float, info: ValidationInfo) -> float:
        """Ensure ask >= bid."""
        values = info.data
        if 'bid' in values and v < values['bid']:
            raise ValueError('ask must be >= bid')
        return v

class TradeData(BaseModel):
    """Validated trade data schema."""
    timestamp: datetime
    instrument: str = Field(..., min_length=1, max_length=50)
    price: float = Field(..., gt=0)
    size: float = Field(..., gt=0)
    side: OrderSide
    trade_id: Optional[str] = Field(None, max_length=100)

class OrderBookLevelData(BaseModel):
    """Validated order book level schema."""
    price: float = Field(..., gt=0)
    size: float = Field(..., ge=0)
    num_orders: Optional[int] = Field(None, ge=1)


class OrderBookData(BaseModel):
    """Validated order book schema."""
    timestamp: datetime
    instrument: str = Field(..., min_length=1, max_length=50)
    bids: List[OrderBookLevelData]
    asks: List[OrderBookLevelData]

    @field_validator('bids', 'asks')
    @classmethod
    def validate_levels(cls, v: List[OrderBookLevelData]) -> List[OrderBookLevelData]:
        """Ensure levels are not empty and sorted."""
        if not v:
            return v
        # Check prices are positive
        for level in v:
            if level.price <= 0:
                raise ValueError('Order book level price must be positive')
        return v

    @model_validator(mode='after')
    def validate_spread(self) -> 'OrderBookData':
        """Ensure best ask > best bid if both exist."""
        bids = self.bids
        asks = self.asks

        if bids and asks:
            best_bid = max(b.price for b in bids)
            best_ask = min(a.price for a in asks)
            if best_ask <= best_bid:
                raise ValueError(f'Invalid order book: best_ask ({best_ask}) <= best_bid ({best_bid})')

        return self

class OptionContractData(BaseModel):
    """Validated option contract schema."""
    underlying: str = Field(..., min_length=1, max_length=10)
    strike: float = Field(..., gt=0)
    expiry: datetime
    option_type: OptionType

    @field_validator('underlying')
    @classmethod
    def validate_underlying(cls, v: str) -> str:
        """Ensure underlying is uppercase."""
        return v.upper()

class GreeksData(BaseModel):
    """Validated Greeks data schema."""
    delta: float = Field(..., ge=-1, le=1)
    gamma: float = Field(..., ge=0)
    theta: float  # Can be negative
    vega: float = Field(..., ge=0)
    rho: float
    iv: Optional[float] = Field(None, ge=0, le=5)  # Implied volatility (0-500%)


class MarketStateData(BaseModel):
    """Validated market state schema."""
    timestamp: datetime
    instrument: str = Field(..., min_length=1, max_length=50)
    spot_price: float = Field(..., gt=0)
    volatility: Optional[float] = Field(None, ge=0, le=5)
    risk_free_rate: float = Field(0.05, ge=-0.1, le=0.5)

class BacktestConfig(BaseModel):
    """Validated backtest configuration."""
    start_date: datetime
    end_date: datetime
    instruments: List[str] = Field(..., min_length=1)
    initial_capital: float = Field(100000.0, gt=0)
    commission_rate: float = Field(0.001, ge=0, le=0.1)
    slippage_bps: float = Field(1.0, ge=0, le=100)
    max_position_size: float = Field(1.0, gt=0, le=10)

    @model_validator(mode='after')
    def validate_dates(self) -> 'BacktestConfig':
        """Ensure end_date > start_date."""
        start = self.start_date
        end = self.end_date
        if end <= start:
            raise ValueError('end_date must be after start_date')
        return self

    @field_validator('instruments')
    @classmethod
    def validate_instruments(cls, v: List[str]) -> List[str]:
        """Ensure instruments are unique and non-empty."""
        if len(v) != len(set(v)):
            raise ValueError('instruments must be unique')
        return v

class DownloadRequest(BaseModel):
    """Validated data download request."""
    exchange: Literal["deribit", "binance"]
    data_type: Literal["trades", "orderbook", "ticks", "ohlcv"]
    instrument: str = Field(..., min_length=1, max_length=50)
    start: datetime
    end: datetime

    @model_validator(mode='after')
    def validate_date_range(self) -> 'DownloadRequest':
        """Ensure date range is valid."""
        start = self.start
        end = self.end
        if start and end:
            if end <= start:
                raise ValueError('end must be after start')
            # Max 1 year range
            if (end - start).days > 365:
                raise ValueError('date range must not exceed 1 year')
        return self

class WebSocketConfig(BaseModel):
    """Validated WebSocket configuration."""
    reconnect_interval: float = Field(5.0, ge=1.0, le=60.0)
    max_reconnects: int = Field(10, ge=1, le=100)
    ping_interval: float = Field(20.0, ge=5.0, le=120.0)
    ping_timeout: float = Field(10.0, ge=1.0, le=60.0)

    @model_validator(mode='after')
    def validate_timeouts(self) -> 'WebSocketConfig':
        """Ensure ping_interval > ping_timeout."""
        interval = self.ping_interval
        timeout = self.ping_timeout
        if timeout >= interval:
            raise ValueError('ping_timeout must be less than ping_interval')
        return self


# Type conversion utilities
def tick_to_schema(tick: "Tick") -> TickData:
    """Convert core.types.Tick to TickData schema."""
    return TickData(
        timestamp=tick.timestamp,
        instrument=tick.instrument,
        bid=tick.bid,
        ask=tick.ask,
        bid_size=tick.bid_size,
        ask_size=tick.ask_size,
    )


def trade_to_schema(trade: "Trade") -> TradeData:
    """Convert core.types.Trade to TradeData schema."""
    return TradeData(
        timestamp=trade.timestamp,
        instrument=trade.instrument,
        price=trade.price,
        size=trade.size,
        side=OrderSide(trade.side.value),
        trade_id=trade.trade_id,
    )


def orderbook_to_schema(ob: "OrderBook") -> OrderBookData:
    """Convert core.types.OrderBook to OrderBookData schema."""
    return OrderBookData(
        timestamp=ob.timestamp,
        instrument=ob.instrument,
        bids=[OrderBookLevelData(price=b.price, size=b.size, num_orders=getattr(b, 'num_orders', None)) for b in ob.bids],
        asks=[OrderBookLevelData(price=a.price, size=a.size, num_orders=getattr(a, 'num_orders', None)) for a in ob.asks],
    )
