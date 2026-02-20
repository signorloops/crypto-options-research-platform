"""
Deribit historical data downloader and API client.
Deribit provides free historical data for research purposes.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from core.exchange.base import ExchangeInterface
from core.types import OptionContract, OptionType, OrderBook, OrderBookLevel, Tick, Trade
from utils.logging_config import get_logger, log_extra

if TYPE_CHECKING:
    from data.cache import DataCache
    from data.streaming import DeribitStream

logger = get_logger(__name__)

MAX_TRADES_PER_REQUEST = 1000
TRADES_PAGE_WINDOW_HOURS = 1
TRADES_RATE_LIMIT_SLEEP_SECONDS = 0.1
SNAPSHOT_RATE_LIMIT_THRESHOLD = 100
SNAPSHOT_RATE_LIMIT_SLEEP_SECONDS = 0.05
TICK_RESAMPLE_FREQUENCY = "1s"


class DeribitAPIError(Exception):
    """Deribit API error."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class DeribitClient(ExchangeInterface):
    """
    Deribit exchange client for market data.
    Documentation: https://docs.deribit.com/
    """

    BASE_URL = os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com/api/v2/public")
    WS_URL = os.getenv("DERIBIT_WS_URL", "wss://www.deribit.com/ws/api/v2")

    def __init__(self):
        super().__init__("deribit")
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_streams: List[DeribitStream] = []

    def reset(self) -> None:
        """Reset client state (used by tests)."""
        self._active_streams.clear()
        # Session lifecycle is managed by disconnect().

    async def connect(self) -> None:
        """Create aiohttp session."""
        self._session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        """Close session and all WebSocket streams."""
        # Close all active WebSocket streams.
        for stream in self._active_streams:
            await stream.disconnect()
        self._active_streams.clear()

        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Make API request."""
        if not self._session:
            await self.connect()

        url = f"{self.BASE_URL}/{method}"
        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            if "error" in data and data["error"]:
                error = data["error"]
                raise DeribitAPIError(
                    message=error.get("message", "Unknown error"), code=error.get("code")
                )
            return data.get("result", {})

    async def get_instruments(
        self, currency: Optional[str] = "BTC", instrument_type: Optional[str] = "option"
    ) -> List[OptionContract]:
        """Get available options."""
        result = await self._request(
            "get_instruments", {"currency": currency, "kind": instrument_type, "expired": "false"}
        )

        contracts = []
        for inst in result:
            contract = OptionContract(
                underlying=inst["base_currency"],
                strike=inst["strike"],
                expiry=datetime.fromtimestamp(inst["expiration_timestamp"] / 1000),
                option_type=OptionType.CALL if inst["option_type"] == "call" else OptionType.PUT,
            )
            contracts.append(contract)

        return contracts

    async def get_order_book(self, instrument: str, depth: int = 10) -> OrderBook:
        """Get order book snapshot."""
        result = await self._request(
            "get_order_book", {"instrument_name": instrument, "depth": depth}
        )

        bids = [OrderBookLevel(price=b[0], size=b[1]) for b in result.get("bids", [])]
        asks = [OrderBookLevel(price=a[0], size=a[1]) for a in result.get("asks", [])]

        return OrderBook(
            timestamp=datetime.now(timezone.utc), instrument=instrument, bids=bids, asks=asks
        )

    async def get_tick(self, instrument: str) -> Tick:
        """Get current best bid/ask."""
        ob = await self.get_order_book(instrument, depth=1)
        return Tick(
            timestamp=ob.timestamp,
            instrument=instrument,
            bid=ob.best_bid or 0,
            ask=ob.best_ask or 0,
            bid_size=ob.bids[0].size if ob.bids else 0,
            ask_size=ob.asks[0].size if ob.asks else 0,
        )

    async def get_ticker(self, instrument: str) -> Dict[str, Any]:
        """
        Get ticker data for an instrument.

        For options, includes Greeks and implied volatility.

        Returns:
            Dict with fields like:
            - mark_price, last_price, bid, ask
            - mark_iv (implied volatility for options)
            - greeks: delta, gamma, theta, vega, rho
            - open_interest, volume_24h
        """
        result = await self._request("ticker", {"instrument_name": instrument})

        return result

    async def get_option_greeks(self, instrument: str) -> Optional[Dict[str, float]]:
        """
        Get Greeks for an option contract.

        Args:
            instrument: Option instrument name (e.g., "BTC-27DEC24-80000-C")

        Returns:
            Dict with delta, gamma, theta, vega, rho, or None if not an option
        """
        ticker = await self.get_ticker(instrument)

        # Check if this is an option (has greeks data)
        if "greeks" not in ticker:
            return None

        greeks = ticker.get("greeks", {})
        return {
            "delta": float(greeks.get("delta", 0)),
            "gamma": float(greeks.get("gamma", 0)),
            "theta": float(greeks.get("theta", 0)),
            "vega": float(greeks.get("vega", 0)),
            "rho": float(greeks.get("rho", 0)),
        }

    async def get_option_iv(self, instrument: str) -> Optional[float]:
        """
        Get implied volatility for an option contract.

        Args:
            instrument: Option instrument name (e.g., "BTC-27DEC24-80000-C")

        Returns:
            Implied volatility as float, or None if not available
        """
        ticker = await self.get_ticker(instrument)
        iv = ticker.get("mark_iv") or ticker.get("iv")

        if iv is not None:
            return float(iv)
        return None

    async def get_volatility_index_history(
        self,
        currency: str = "BTC",
        resolution: str = "1D",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get volatility index (DVOL) historical data.

        Deribit Volatility Index (DVOL) is similar to VIX.

        Args:
            currency: 'BTC' or 'ETH'
            resolution: '1D', '1H', etc.
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with timestamp, open, high, low, close
        """
        params: Dict[str, Any] = {"currency": currency, "resolution": resolution}

        if start:
            params["start_timestamp"] = int(start.timestamp() * 1000)
        if end:
            params["end_timestamp"] = int(end.timestamp() * 1000)

        result = await self._request("get_volatility_index_data", params)

        data = result.get("data", [])
        if not data:
            return pd.DataFrame()

        # Data format: [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float) / 100  # Convert from percentage

        return df.sort_values("timestamp").reset_index(drop=True)

    async def get_trades(
        self, instrument: str, start: datetime, end: datetime, limit: int = MAX_TRADES_PER_REQUEST
    ) -> List[Trade]:
        """
        Get historical trades.
        Note: Deribit has rate limits, may need to paginate for large requests.
        """
        from core.types import OrderSide

        trades = []
        current_start = start

        while current_start < end:
            result = await self._request(
                "get_last_trades_by_instrument",
                {
                    "instrument_name": instrument,
                    "start_timestamp": int(current_start.timestamp() * 1000),
                    "end_timestamp": int(
                        min(
                            current_start + timedelta(hours=TRADES_PAGE_WINDOW_HOURS), end
                        ).timestamp()
                        * 1000
                    ),
                    "count": min(limit, MAX_TRADES_PER_REQUEST),
                },
            )

            batch = result.get("trades", [])
            if not batch:
                break

            for t in batch:
                trades.append(
                    Trade(
                        timestamp=datetime.fromtimestamp(t["timestamp"] / 1000),
                        instrument=instrument,
                        price=t["price"],
                        size=t["amount"],
                        side=OrderSide.BUY if t["direction"] == "buy" else OrderSide.SELL,
                        trade_id=t.get("trade_id"),
                    )
                )

            # Move to next batch
            last_time = batch[-1]["timestamp"]
            current_start = datetime.fromtimestamp(last_time / 1000, tz=timezone.utc) + timedelta(
                milliseconds=1
            )

            # Rate limiting
            await asyncio.sleep(TRADES_RATE_LIMIT_SLEEP_SECONDS)

        return trades

    async def get_historical_volatility(
        self, currency: str = "BTC", period_days: int = 30
    ) -> float:
        """Get Deribit calculated historical volatility."""
        result = await self._request("get_historical_volatility", {"currency": currency})

        # Result is list of [timestamp, vol] pairs, get most recent
        if result and len(result) > 0:
            return result[-1][1] / 100  # Convert from percentage
        return 0.5  # Default 50%

    async def subscribe_order_book(
        self, instruments: List[str], callback: Callable[[OrderBook], None]
    ) -> None:
        """
        WebSocket subscription for real-time order book updates.

        Args:
            instruments: List of instrument names (e.g., ["BTC-27DEC24-80000-C"])
            callback: Function called with OrderBook updates
        """
        from data.streaming import DeribitStream

        stream = DeribitStream()
        self._active_streams.append(stream)
        stream.add_callback("orderbook", callback)
        await stream.connect(instruments)

    async def subscribe_trades(
        self, instruments: List[str], callback: Callable[[Trade], None]
    ) -> None:
        """
        WebSocket subscription for real-time trade updates.

        Args:
            instruments: List of instrument names
            callback: Function called with Trade updates
        """
        from data.streaming import DeribitStream

        stream = DeribitStream()
        self._active_streams.append(stream)
        stream.add_callback("trade", callback)
        await stream.connect(instruments)


class DeribitDataDownloader:
    """
    High-level data downloader with integrated caching.
    """

    def __init__(self, cache: Optional["DataCache"] = None):
        from data.cache import DataCache

        self.client = DeribitClient()
        self.cache = cache or DataCache()

    @staticmethod
    def _recent_cache_bounds(hours: int = 24) -> Tuple[datetime, datetime]:
        """Return a bounded recent time window for point-in-time cache lookups."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return start, end

    async def download_trades(
        self, instrument: str, start: datetime, end: datetime, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download historical trades with automatic caching.

        Args:
            instrument: Option contract name (e.g., "BTC-27DEC24-80000-C")
            start: Start datetime
            end: End datetime
            use_cache: Whether to use cache

        Returns:
            DataFrame with columns: timestamp, price, size, side
        """
        # Check cache
        if use_cache:
            cached = self.cache.get("deribit", "trades", instrument, start, end)
            if cached is not None:
                logger.info(
                    "Using cached data", extra=log_extra(instrument=instrument, records=len(cached))
                )
                return cached

        # Download
        logger.info("Downloading trades from Deribit", extra=log_extra(instrument=instrument))
        async with self.client:
            trades = await self.client.get_trades(instrument, start, end)

        df = pd.DataFrame(
            [
                {"timestamp": t.timestamp, "price": t.price, "size": t.size, "side": t.side.value}
                for t in trades
            ]
        )

        # Cache result
        if use_cache and len(df) > 0:
            self.cache.put_range(df, "deribit", "trades", instrument)
            logger.info("Cached trades", extra=log_extra(instrument=instrument, count=len(df)))

        return df

    async def download_order_book_snapshots(
        self, instrument: str, timestamps: List[datetime], use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download order book snapshots at specific times.
        """
        snapshots = []

        # Check cache for existing data
        if use_cache and len(timestamps) > 0:
            start, end = min(timestamps), max(timestamps)
            if self.cache.exists("deribit", "orderbook", instrument, start, end):
                cached = self.cache.get("deribit", "orderbook", instrument, start, end)
                if cached is not None:
                    return cached

        logger.info("Downloading order book snapshots", extra=log_extra(count=len(timestamps)))

        async with self.client:
            for ts in timestamps:
                ob = await self.client.get_order_book(instrument)
                snapshots.append(
                    {
                        "timestamp": ts,
                        "best_bid": ob.best_bid,
                        "best_ask": ob.best_ask,
                        "bid_1_size": ob.bids[0].size if ob.bids else 0,
                        "ask_1_size": ob.asks[0].size if ob.asks else 0,
                        "spread": ob.spread,
                        "imbalance": ob.imbalance(),
                        "bid_volume_5": sum(lvl.size for lvl in ob.bids[:5]),
                        "ask_volume_5": sum(lvl.size for lvl in ob.asks[:5]),
                    }
                )

                if len(timestamps) > SNAPSHOT_RATE_LIMIT_THRESHOLD:
                    await asyncio.sleep(SNAPSHOT_RATE_LIMIT_SLEEP_SECONDS)

        df = pd.DataFrame(snapshots)

        if use_cache and len(df) > 0:
            self.cache.put_range(df, "deribit", "orderbook", instrument)

        return df

    async def download_tick_data(
        self, instrument: str, start: datetime, end: datetime, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download tick-level data (best bid/ask over time).
        """
        if use_cache:
            cached = self.cache.get("deribit", "ticks", instrument, start, end)
            if cached is not None:
                return cached

        # Deribit doesn't have direct tick endpoint, so we construct from trades
        trades_df = await self.download_trades(instrument, start, end, use_cache)

        if len(trades_df) == 0:
            return pd.DataFrame()

        # Aggregate to 1-second ticks
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        trades_df.set_index("timestamp", inplace=True)

        # Resample to get OHLC-style ticks
        tick_df = trades_df.resample(TICK_RESAMPLE_FREQUENCY).agg(
            {"price": ["first", "last", "max", "min"], "size": "sum"}
        )

        tick_df.columns = ["open", "close", "high", "low", "volume"]
        tick_df = tick_df.dropna()

        if use_cache and len(tick_df) > 0:
            self.cache.put_range(tick_df.reset_index(), "deribit", "ticks", instrument)

        return tick_df.reset_index()

    async def download_greeks(
        self, instrument: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download option Greeks data.

        Args:
            instrument: Option instrument name (e.g., "BTC-27DEC24-80000-C")
            use_cache: Whether to use cache

        Returns:
            DataFrame with Greeks or None if not an option
        """
        if use_cache:
            start, end = self._recent_cache_bounds(hours=24)
            cached = self.cache.get("deribit", "greeks", instrument, start, end)
            if cached is not None and not cached.empty:
                return cached

        async with self.client:
            greeks = await self.client.get_option_greeks(instrument)

        if greeks is None:
            return None

        df = pd.DataFrame([greeks])
        df["instrument"] = instrument
        now = datetime.now(timezone.utc)
        df["timestamp"] = now

        if use_cache:
            self.cache.put(df, "deribit", "greeks", instrument, now)

        return df

    async def download_iv(self, instrument: str, use_cache: bool = True) -> Optional[float]:
        """
        Download implied volatility for an option.

        Args:
            instrument: Option instrument name
            use_cache: Whether to use cache

        Returns:
            Implied volatility or None
        """
        cache_key = f"{instrument}_iv"
        if use_cache:
            start, end = self._recent_cache_bounds(hours=24)
            cached = self.cache.get("deribit", "iv", cache_key, start, end)
            if cached is not None and not cached.empty and "iv" in cached.columns:
                return float(cached["iv"].iloc[-1])

        async with self.client:
            iv = await self.client.get_option_iv(instrument)

        if use_cache and iv is not None:
            now = datetime.now(timezone.utc)
            df = pd.DataFrame([{"iv": iv, "timestamp": now}])
            self.cache.put(df, "deribit", "iv", cache_key, now)

        return iv

    async def download_volatility_index(
        self, currency: str = "BTC", period_days: int = 30, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download DVOL (Deribit Volatility Index) history.

        Args:
            currency: 'BTC' or 'ETH'
            period_days: Number of days of history
            use_cache: Whether to use cache

        Returns:
            DataFrame with DVOL history
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=period_days)

        if use_cache:
            cached = self.cache.get("deribit", "dvol", currency, start, end)
            if cached is not None and not cached.empty:
                return cached

        logger.info("Downloading DVOL data", extra=log_extra(currency=currency))

        async with self.client:
            df = await self.client.get_volatility_index_history(
                currency=currency, resolution="1D", start=start, end=end
            )

        if use_cache and not df.empty:
            self.cache.put_range(df, "deribit", "dvol", currency)

        return df

    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        return self.cache.get_cache_info()

    def list_cached_instruments(self, data_type: str = "trades") -> List[str]:
        """List all instruments with cached data."""
        root = Path(self.cache.raw_dir) / "deribit" / data_type
        if not root.exists():
            return []

        return sorted(path.name.replace("_", "-") for path in root.iterdir() if path.is_dir())
