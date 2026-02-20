"""
Base exchange interface for unified market data access.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, List, Optional

from core.types import OptionContract, OrderBook, Tick, Trade


class ExchangeInterface(ABC):
    """
    Abstract base class for exchange implementations.
    Provides unified interface for different exchanges.
    """

    def __init__(self, name: str):
        self.name = name
        self._running = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def get_instruments(
        self,
        currency: Optional[str] = None,
        instrument_type: Optional[str] = None
    ) -> List[OptionContract]:
        """
        Get list of available instruments.

        Args:
            currency: Filter by currency (BTC, ETH, etc.)
            instrument_type: Filter by type (option, future, etc.)

        Returns:
            List of option contracts
        """
        pass

    @abstractmethod
    async def get_order_book(
        self,
        instrument: str,
        depth: int = 10
    ) -> OrderBook:
        """
        Get current order book.

        Args:
            instrument: Instrument name
            depth: Number of levels to fetch

        Returns:
            OrderBook snapshot
        """
        pass

    @abstractmethod
    async def get_tick(self, instrument: str) -> Tick:
        """Get current best bid/ask."""
        pass

    @abstractmethod
    async def get_trades(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        limit: int = 1000
    ) -> List[Trade]:
        """
        Get historical trades.

        Args:
            instrument: Instrument name
            start: Start time
            end: End time
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        pass

    @abstractmethod
    async def get_historical_volatility(
        self,
        currency: str,
        period_days: int = 30
    ) -> float:
        """Get historical realized volatility."""
        pass

    @abstractmethod
    async def subscribe_order_book(
        self,
        instruments: List[str],
        callback: Callable[[OrderBook], None]
    ) -> None:
        """Subscribe to real-time order book updates."""
        pass

    @abstractmethod
    async def subscribe_trades(
        self,
        instruments: List[str],
        callback: Callable[[Trade], None]
    ) -> None:
        """Subscribe to real-time trade updates."""
        pass

    async def __aenter__(self) -> "ExchangeInterface":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        await self.disconnect()
