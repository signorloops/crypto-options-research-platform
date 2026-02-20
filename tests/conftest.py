"""
Pytest fixtures and configuration.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

# Ensure the project root is first on sys.path so local `data` package wins.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# If an external namespace package named `data` was preloaded, clear it.
if "data" in sys.modules and getattr(sys.modules["data"], "__file__", None) is None:
    del sys.modules["data"]
if "data.cache" in sys.modules and getattr(sys.modules["data.cache"], "__file__", None) is None:
    del sys.modules["data.cache"]

from core.types import (
    Greeks,
    OptionContract,
    OptionType,
    OrderBook,
    OrderBookLevel,
    OrderSide,
    Position,
    Tick,
    Trade,
)
from data.generators.synthetic import CompleteMarketSimulator


@pytest.fixture
def sample_order_book():
    """Create a sample order book for testing."""
    return OrderBook(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        instrument="BTC-PERPETUAL",
        bids=[
            OrderBookLevel(price=50000, size=1.5, num_orders=5),
            OrderBookLevel(price=49990, size=2.0, num_orders=3),
            OrderBookLevel(price=49980, size=1.0, num_orders=2),
        ],
        asks=[
            OrderBookLevel(price=50010, size=1.0, num_orders=4),
            OrderBookLevel(price=50020, size=2.5, num_orders=6),
            OrderBookLevel(price=50030, size=1.5, num_orders=2),
        ]
    )


@pytest.fixture
def sample_tick():
    """Create a sample tick for testing."""
    return Tick(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        instrument="BTC-PERPETUAL",
        bid=50000,
        ask=50010,
        bid_size=1.5,
        ask_size=1.0
    )


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    return [
        Trade(timestamp=base_time + timedelta(seconds=i),
              instrument="BTC-PERPETUAL",
              price=50000 + i * 10,
              size=0.1 * (i + 1),
              side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL)
        for i in range(10)
    ]


@pytest.fixture
def sample_option_contract():
    """Create a sample option contract (coin-margined)."""
    return OptionContract(
        underlying="BTC-USD",
        strike=50000,
        expiry=datetime(2024, 12, 31),
        option_type=OptionType.CALL,
        inverse=True
    )


@pytest.fixture
def sample_greeks():
    """Create sample Greeks."""
    return Greeks(
        delta=0.5,
        gamma=0.001,
        theta=-10.0,
        vega=5.0,
        rho=0.1,
        vanna=0.01,
        charm=-0.001
    )


@pytest.fixture
def sample_position():
    """Create a sample position."""
    return Position(
        instrument="BTC-PERPETUAL",
        size=1.5,
        avg_entry_price=49500
    )


@pytest.fixture(scope="session")
def sample_market_data():
    """Generate sample market data for testing."""
    sim = CompleteMarketSimulator(seed=42)
    # Use fewer hours for faster tests while maintaining coverage
    return sim.generate(hours=2, include_options=False)


@pytest.fixture
def empty_order_book():
    """Create an empty order book for edge case testing."""
    return OrderBook(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        instrument="BTC-PERPETUAL",
        bids=[],
        asks=[]
    )


@pytest.fixture
def deep_order_book():
    """Create a deep order book for testing."""
    bids = [OrderBookLevel(price=50000 - i * 10, size=1.0 + i * 0.1) for i in range(20)]
    asks = [OrderBookLevel(price=50010 + i * 10, size=1.0 + i * 0.1) for i in range(20)]

    return OrderBook(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        instrument="BTC-PERPETUAL",
        bids=bids,
        asks=asks
    )
