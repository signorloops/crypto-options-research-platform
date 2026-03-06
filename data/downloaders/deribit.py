"""
Deribit historical data downloader and API client.
Deribit provides free historical data for research purposes.
"""

from __future__ import annotations

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


def _cached_frame_or_none(
    *,
    cache: "DataCache",
    exchange: str,
    data_type: str,
    instrument: str,
    start: datetime,
    end: datetime,
    use_cache: bool,
) -> Optional[pd.DataFrame]:
    if not use_cache:
        return None
    return cache.get(exchange, data_type, instrument, start, end)


def _cache_range_if_nonempty(
    *,
    cache: "DataCache",
    df: pd.DataFrame,
    exchange: str,
    data_type: str,
    instrument: str,
    use_cache: bool,
) -> None:
    if use_cache and df is not None and not df.empty:
        cache.put_range(df, exchange, data_type, instrument)


def _recent_cached_nonempty(
    *,
    cache: "DataCache",
    exchange: str,
    data_type: str,
    instrument: str,
    now: datetime,
    hours: int,
    use_cache: bool,
) -> Optional[pd.DataFrame]:
    if not use_cache:
        return None
    start = now - timedelta(hours=hours)
    cached = cache.get(exchange, data_type, instrument, start, now)
    if cached is None or cached.empty:
        return None
    return cached


def _greeks_dataframe(greeks: Dict[str, Any], instrument: str, now: datetime) -> pd.DataFrame:
    df = pd.DataFrame([greeks])
    df["instrument"] = instrument
    df["timestamp"] = now
    return df


def _cached_greeks_frame(
    *,
    cache: "DataCache",
    instrument: str,
    now: datetime,
    use_cache: bool,
) -> Optional[pd.DataFrame]:
    return _recent_cached_nonempty(
        cache=cache,
        exchange="deribit",
        data_type="greeks",
        instrument=instrument,
        now=now,
        hours=24,
        use_cache=use_cache,
    )


async def _fetch_greeks_frame(
    *,
    client: "DeribitClient",
    instrument: str,
    now: datetime,
) -> Optional[pd.DataFrame]:
    async with client:
        greeks = await client.get_option_greeks(instrument)
    if greeks is None:
        return None
    return _greeks_dataframe(greeks, instrument, now)


def _cache_greeks_frame_if_enabled(
    *,
    cache: "DataCache",
    df: pd.DataFrame,
    instrument: str,
    now: datetime,
    use_cache: bool,
) -> None:
    if use_cache:
        cache.put(df, "deribit", "greeks", instrument, now)


async def _download_greeks_with_cache(
    *,
    cache: "DataCache",
    client: "DeribitClient",
    instrument: str,
    use_cache: bool,
) -> Optional[pd.DataFrame]:
    now = datetime.now(timezone.utc)
    cached = _cached_greeks_frame(
        cache=cache,
        instrument=instrument,
        now=now,
        use_cache=use_cache,
    )
    if cached is not None:
        return cached
    df = await _fetch_greeks_frame(
        client=client,
        instrument=instrument,
        now=now,
    )
    if df is None:
        return None
    _cache_greeks_frame_if_enabled(
        cache=cache,
        df=df,
        instrument=instrument,
        now=now,
        use_cache=use_cache,
    )
    return df


def _dvol_time_window(period_days: int) -> Tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=period_days)
    return start, end


def _trades_to_ticks(trades_df: pd.DataFrame) -> pd.DataFrame:
    if len(trades_df) == 0:
        return pd.DataFrame()
    ticks = trades_df.copy()
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"])
    ticks.set_index("timestamp", inplace=True)
    tick_df = ticks.resample(TICK_RESAMPLE_FREQUENCY).agg(
        {"price": ["first", "last", "max", "min"], "size": "sum"}
    )
    tick_df.columns = ["open", "close", "high", "low", "volume"]
    tick_df = tick_df.dropna()
    return tick_df.reset_index()


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
                expiry=datetime.fromtimestamp(inst["expiration_timestamp"] / 1000, tz=timezone.utc),
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
        """Get Deribit volatility index (DVOL) historical OHLC data."""
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
        trades = []
        current_start = start

        while current_start < end:
            params = self._build_trade_query_params(
                instrument=instrument,
                current_start=current_start,
                end=end,
                limit=limit,
            )
            result = await self._request(
                "get_last_trades_by_instrument",
                params,
            )

            batch = result.get("trades", [])
            if not batch:
                break
            trades.extend(self._parse_trade_batch(batch=batch, instrument=instrument))
            current_start = self._advance_trade_cursor(batch)
            await asyncio.sleep(TRADES_RATE_LIMIT_SLEEP_SECONDS)

        return trades

    @staticmethod
    def _build_trade_query_params(
        *, instrument: str, current_start: datetime, end: datetime, limit: int
    ) -> Dict[str, int | str]:
        window_end = min(current_start + timedelta(hours=TRADES_PAGE_WINDOW_HOURS), end)
        return {
            "instrument_name": instrument,
            "start_timestamp": int(current_start.timestamp() * 1000),
            "end_timestamp": int(window_end.timestamp() * 1000),
            "count": min(limit, MAX_TRADES_PER_REQUEST),
        }

    @staticmethod
    def _parse_trade_batch(*, batch: List[Dict[str, Any]], instrument: str) -> List[Trade]:
        from core.types import OrderSide

        return [
            Trade(
                timestamp=datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc),
                instrument=instrument,
                price=t["price"],
                size=t["amount"],
                side=OrderSide.BUY if t["direction"] == "buy" else OrderSide.SELL,
                trade_id=t.get("trade_id"),
            )
            for t in batch
        ]

    @staticmethod
    def _advance_trade_cursor(batch: List[Dict[str, Any]]) -> datetime:
        last_time = batch[-1]["timestamp"]
        return datetime.fromtimestamp(last_time / 1000, tz=timezone.utc) + timedelta(milliseconds=1)

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
        """Download historical trades with optional cache lookup/storage."""
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
        """Download order book snapshots at specific times."""
        if (
            cached := self._get_cached_order_book_snapshots(
                instrument=instrument, timestamps=timestamps, use_cache=use_cache
            )
        ) is not None:
            return cached
        snapshots = []
        logger.info("Downloading order book snapshots", extra=log_extra(count=len(timestamps)))
        should_throttle = len(timestamps) > SNAPSHOT_RATE_LIMIT_THRESHOLD
        async with self.client:
            for ts in timestamps:
                ob = await self.client.get_order_book(instrument)
                snapshots.append(self._snapshot_row(ts, ob))
                if should_throttle:
                    await asyncio.sleep(SNAPSHOT_RATE_LIMIT_SLEEP_SECONDS)
        df = pd.DataFrame(snapshots)
        if use_cache and len(df) > 0:
            self.cache.put_range(df, "deribit", "orderbook", instrument)
        return df

    def _get_cached_order_book_snapshots(
        self, *, instrument: str, timestamps: List[datetime], use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """Try reading order-book snapshots from cache for the requested window."""
        if not use_cache or not timestamps:
            return None
        start, end = min(timestamps), max(timestamps)
        if not self.cache.exists("deribit", "orderbook", instrument, start, end):
            return None
        return self.cache.get("deribit", "orderbook", instrument, start, end)

    @staticmethod
    def _snapshot_row(ts: datetime, ob: OrderBook) -> Dict[str, Any]:
        """Convert one order-book snapshot into a tabular row."""
        return {
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

    async def download_tick_data(
        self, instrument: str, start: datetime, end: datetime, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download tick-level data (best bid/ask over time).
        """
        cached = _cached_frame_or_none(
            cache=self.cache,
            exchange="deribit",
            data_type="ticks",
            instrument=instrument,
            start=start,
            end=end,
            use_cache=use_cache,
        )
        if cached is not None:
            return cached

        # Deribit doesn't have direct tick endpoint, so we construct from trades
        trades_df = await self.download_trades(instrument, start, end, use_cache)
        tick_df = _trades_to_ticks(trades_df)
        _cache_range_if_nonempty(
            cache=self.cache,
            df=tick_df,
            exchange="deribit",
            data_type="ticks",
            instrument=instrument,
            use_cache=use_cache,
        )
        return tick_df

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
        return await _download_greeks_with_cache(
            cache=self.cache,
            client=self.client,
            instrument=instrument,
            use_cache=use_cache,
        )

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
        """Download DVOL (Deribit volatility index) history."""
        start, end = _dvol_time_window(period_days)
        cached = _cached_frame_or_none(
            cache=self.cache,
            exchange="deribit",
            data_type="dvol",
            instrument=currency,
            start=start,
            end=end,
            use_cache=use_cache,
        )
        if cached is not None and not cached.empty:
            return cached

        logger.info("Downloading DVOL data", extra=log_extra(currency=currency))
        async with self.client:
            df = await self.client.get_volatility_index_history(
                currency=currency, resolution="1D", start=start, end=end
            )
        _cache_range_if_nonempty(
            cache=self.cache,
            df=df,
            exchange="deribit",
            data_type="dvol",
            instrument=currency,
            use_cache=use_cache,
        )

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
