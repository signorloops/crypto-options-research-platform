"""
OKX API client for market data (coin-margined options only).
Supports spot, futures, perpetual swaps, and coin-margined options (BTC-USD, ETH-USD).
Documentation: https://www.okx.com/docs-v5/en/#rest-api-public-data
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd

from core.exchange.base import ExchangeInterface
from core.types import OptionContract, OptionType, OrderBook, OrderBookLevel, OrderSide, Tick, Trade
from utils.logging_config import get_logger, log_extra

if TYPE_CHECKING:
    from data.streaming import OKXStream

logger = get_logger(__name__)


class OKXAPIError(Exception):
    """OKX API error."""

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


class OKXClient(ExchangeInterface):
    """
    OKX exchange client for market data (coin-margined options only).
    Only supports BTC-USD and ETH-USD coin-margined options.
    """

    BASE_URL = os.getenv("OKX_BASE_URL", "https://www.okx.com")
    WS_URL = os.getenv("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public")

    # Valid coin-margined underlyings
    VALID_UNDERLYINGS = {"BTC-USD", "ETH-USD"}

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__("okx")
        self.api_key = api_key
        self.api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_streams: List["OKXStream"] = []

    def reset(self) -> None:
        """Reset client state for testing."""
        self._active_streams.clear()

    async def connect(self) -> None:
        """Create aiohttp session."""
        self._session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        """Close session and all WebSocket streams."""
        for stream in self._active_streams:
            await stream.disconnect()
        self._active_streams.clear()

        if self._session:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: float = 30.0
    ) -> Dict:
        """Make API request with timeout.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            timeout: Request timeout in seconds (default: 30)

        Returns:
            API response data

        Raises:
            OKXAPIError: If API returns error
            asyncio.TimeoutError: If request times out
        """
        if not self._session:
            await self.connect()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {}
        if self.api_key:
            headers["OK-ACCESS-KEY"] = self.api_key

        try:
            async with self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if data.get("code") != "0":
                    raise OKXAPIError(
                        data.get("msg", "Unknown error"),
                        code=data.get("code")
                    )
                return data
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {endpoint}")
            raise

    async def get_instruments(
        self,
        currency: Optional[str] = None,
        instrument_type: str = "SPOT"
    ) -> List[str]:
        """
        Get available trading instruments.

        Args:
            currency: Base currency (e.g., 'BTC', 'ETH')
            instrument_type: 'SPOT', 'SWAP', 'FUTURES', 'OPTION'
        """
        params = {"instType": instrument_type}

        # For options, only support coin-margined (BTC-USD, ETH-USD)
        if instrument_type == "OPTION" and currency:
            params["uly"] = f"{currency}-USD"

        result = await self._request("/api/v5/public/instruments", params)

        instruments = []
        for inst in result.get("data", []):
            if inst.get("state") == "live":
                instruments.append(inst["instId"])

        return instruments

    async def get_option_instruments(
        self,
        underlying: str = "BTC-USD"
    ) -> List[OptionContract]:
        """
        Get option instruments with detailed info (coin-margined only).

        Args:
            underlying: 'BTC-USD' or 'ETH-USD' (coin-margined only)
        """
        # Validate: only support coin-margined underlyings
        if underlying not in self.VALID_UNDERLYINGS:
            raise ValueError(
                f"Only coin-margined options supported {self.VALID_UNDERLYINGS}, got: {underlying}"
            )
        params = {"instType": "OPTION", "uly": underlying}
        result = await self._request("/api/v5/public/instruments", params)

        contracts = []
        for inst in result.get("data", []):
            if inst.get("state") != "live":
                continue

            # Parse instrument ID: e.g., "BTC-USD-240628-50000-C"
            parts = inst["instId"].split("-")
            if len(parts) != 5:
                continue

            base, quote, expiry_str, strike_str, opt_type = parts

            # Parse expiry
            try:
                expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
            except ValueError:
                continue

            # Parse strike
            try:
                strike = float(strike_str)
            except ValueError:
                continue

            contract = OptionContract(
                underlying=f"{base}-{quote}",
                strike=strike,
                expiry=expiry,
                option_type=OptionType.CALL if opt_type == "C" else OptionType.PUT,
                exchange="okx",
                symbol=inst["instId"],
                inverse=True,  # Coin-margined
                lot_size=float(inst.get("lotSz", 1)),
                tick_size=float(inst.get("tickSz", 0.01))
            )
            contracts.append(contract)

        return contracts

    async def get_order_book(
        self,
        instrument: str,
        depth: int = 10
    ) -> OrderBook:
        """Get order book snapshot."""
        result = await self._request(
            "/api/v5/market/books",
            {"instId": instrument, "sz": min(depth, 400)}
        )

        data = result.get("data", [{}])[0]

        bids = [
            OrderBookLevel(price=float(b[0]), size=float(b[1]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a[0]), size=float(a[1]))
            for a in data.get("asks", [])
        ]

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument=instrument,
            bids=bids,
            asks=asks
        )

    async def get_ticker(self, instrument: str) -> Tick:
        """Get latest ticker data."""
        result = await self._request(
            "/api/v5/market/ticker",
            {"instId": instrument}
        )

        data = result.get("data", [{}])[0]

        return Tick(
            timestamp=datetime.now(timezone.utc),
            instrument=instrument,
            bid=float(data.get("bidPx", 0)),
            ask=float(data.get("askPx", 0)),
            bid_size=float(data.get("bidSz", 0)),
            ask_size=float(data.get("askSz", 0))
        )

    async def get_klines(
        self,
        instrument: str,
        interval: str = "1H",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get OHLCV history.

        Args:
            interval: '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'
        """
        params = {
            "instId": instrument,
            "bar": interval,
            "limit": min(limit, 300)
        }

        if start:
            params["after"] = str(int(start.timestamp() * 1000))
        if end:
            params["before"] = str(int(end.timestamp() * 1000))

        result = await self._request("/api/v5/market/history-candles", params)

        data = result.get("data", [])
        if not data:
            return pd.DataFrame()

        # OKX format: [timestamp, open, high, low, close, vol, volCcy]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "vol_ccy"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df.sort_values("timestamp").reset_index(drop=True)

    async def get_index_price(self, underlying: str) -> float:
        """Get index price for underlying (e.g., 'BTC-USD')."""
        result = await self._request(
            "/api/v5/market/index-tickers",
            {"instId": underlying}
        )

        data = result.get("data", [{}])[0]
        return float(data.get("idxPx", 0))

    async def get_option_market_data(
        self,
        underlying: str = "BTC-USD"
    ) -> List[Dict]:
        """
        Get option market data including IV, Greeks (coin-margined only).

        Returns list of option instruments with:
        - markPrice, markVol (IV)
        - delta, gamma, theta, vega
        - openInterest, volume24h
        """
        # Validate: only support coin-margined underlyings
        if underlying not in self.VALID_UNDERLYINGS:
            raise ValueError(
                f"Only coin-margined options supported {self.VALID_UNDERLYINGS}, got: {underlying}"
            )

        params = {"uly": underlying, "exp": ""}  # Empty exp gets all expiries
        result = await self._request("/api/v5/public/opt-summary", params)

        return result.get("data", [])

    # Implementation of abstract methods from ExchangeInterface

    async def get_tick(self, instrument: str) -> Tick:
        """Get current best bid/ask (alias for get_ticker)."""
        return await self.get_ticker(instrument)

    async def get_trades(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        limit: int = 1000
    ) -> List[Trade]:
        """Get historical trades."""
        params = {
            "instId": instrument,
            "limit": min(limit, 100),
            "begin": str(int(start.timestamp() * 1000)),
            "end": str(int(end.timestamp() * 1000))
        }

        result = await self._request("/api/v5/market/history-trades", params)

        trades = []
        for t in result.get("data", []):
            trades.append(Trade(
                timestamp=datetime.fromtimestamp(int(t["ts"]) / 1000, tz=timezone.utc),
                instrument=instrument,
                price=float(t["px"]),
                size=float(t["sz"]),
                side=OrderSide.BUY if t.get("side") == "buy" else OrderSide.SELL,
                trade_id=t.get("tradeId", "")
            ))

        return trades

    async def get_historical_volatility(
        self,
        currency: str,
        period_days: int = 30
    ) -> float:
        """Get historical realized volatility from index price."""
        underlying = f"{currency}-USD"
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=period_days)

        df = await self.get_klines(underlying, interval="1D", start=start, end=end)

        if len(df) < 2:
            return 0.0

        returns = df["close"].pct_change().dropna()
        if len(returns) < 2:
            return 0.0

        # Annualized volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(365)

        return float(annual_vol)

    async def get_option_volatility_history(
        self,
        underlying: str = "BTC-USD",
        period_days: int = 30
    ) -> pd.DataFrame:
        """
        Get option implied volatility history from OKX.

        OKX provides IV index data via /api/v5/market/index-candles
        for the underlying's option-implied volatility.

        Args:
            underlying: 'BTC-USD' or 'ETH-USD'
            period_days: Number of days of history

        Returns:
            DataFrame with columns: timestamp, open, high, low, close (IV values)
        """
        if underlying not in self.VALID_UNDERLYINGS:
            raise ValueError(
                f"Only coin-margined options supported {self.VALID_UNDERLYINGS}, got: {underlying}"
            )

        # OKX IV index symbol format: BTC-USD-IV (not BTC-USD-IV-INDEX)
        iv_index = f"{underlying}-IV"

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=period_days)

        params = {
            "instId": iv_index,
            "bar": "1D",
            "limit": min(period_days, 100),
            # OKX API: before = end time (get data before this timestamp)
            #           after = start time (get data after this timestamp)
            "before": str(int(end.timestamp() * 1000)),
            "after": str(int(start.timestamp() * 1000))
        }

        try:
            result = await self._request("/api/v5/market/index-candles", params)
            data = result.get("data", [])

            if not data:
                logger.warning(f"No IV data available for {underlying}")
                return pd.DataFrame()

            # Format: [timestamp, open, high, low, close]
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)

            return df.sort_values("timestamp").reset_index(drop=True)

        except OKXAPIError as e:
            logger.error(f"Failed to get IV history: {e}")
            return pd.DataFrame()

    async def get_current_iv_term_structure(
        self,
        underlying: str = "BTC-USD"
    ) -> pd.DataFrame:
        """
        Get current implied volatility term structure by expiry.

        Uses option market data to extract ATM IV for each expiry.

        Returns:
            DataFrame with columns: expiry, days_to_expiry, atm_iv
        """
        from datetime import datetime

        market_data = await self.get_option_market_data(underlying)

        if not market_data:
            return pd.DataFrame()

        # Group by expiry and find ATM options
        expiry_data = {}
        for opt in market_data:
            expiry_ts = opt.get("expTime")
            strike = float(opt.get("stk", 0))
            mark_vol = float(opt.get("markVol", 0))
            uly_px = float(opt.get("ulyPx", 0))  # Current underlying price

            if not expiry_ts or not mark_vol:
                continue

            expiry = datetime.fromtimestamp(int(expiry_ts) / 1000, tz=timezone.utc)
            days_to_exp = (expiry - datetime.now(timezone.utc)).total_seconds() / (24 * 3600)

            if days_to_exp <= 0:
                continue

            # Find options near ATM (within 5%)
            if strike > 0 and uly_px > 0:
                moneyness = abs(strike - uly_px) / uly_px
                if moneyness < 0.05:  # Within 5% of ATM
                    key = expiry_ts
                    if key not in expiry_data or moneyness < expiry_data[key]["moneyness"]:
                        expiry_data[key] = {
                            "expiry": expiry,
                            "days_to_expiry": days_to_exp,
                            "atm_iv": mark_vol,
                            "moneyness": moneyness,
                            "strike": strike
                        }

        if not expiry_data:
            return pd.DataFrame()

        df = pd.DataFrame(list(expiry_data.values()))
        return df.sort_values("days_to_expiry").drop(columns=["moneyness"]).reset_index(drop=True)

    async def subscribe_order_book(
        self,
        instruments: List[str],
        callback: Callable[[OrderBook], None]
    ) -> None:
        """Subscribe to real-time order book updates via WebSocket."""
        from data.streaming import OKXStream

        stream = OKXStream(instruments, order_book_callback=callback)
        try:
            await stream.connect()
            self._active_streams.append(stream)
        except Exception:
            await stream.disconnect()
            raise

    async def subscribe_trades(
        self,
        instruments: List[str],
        callback: Callable[[Trade], None]
    ) -> None:
        """Subscribe to real-time trade updates via WebSocket."""
        from data.streaming import OKXStream

        stream = OKXStream(instruments, trade_callback=callback)
        try:
            await stream.connect()
            self._active_streams.append(stream)
        except Exception:
            await stream.disconnect()
            raise
