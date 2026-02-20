from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.types import OrderBook, OrderBookLevel
from data.cache import DataCache
from data.downloaders.deribit import DeribitDataDownloader


@dataclass
class _StubClient:
    order_book: OrderBook
    greeks_payload: dict[str, float]
    iv_payload: float

    async def __aenter__(self) -> "_StubClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def get_order_book(self, instrument: str) -> OrderBook:
        return self.order_book

    async def get_option_greeks(self, instrument: str) -> dict[str, float]:
        return self.greeks_payload

    async def get_option_iv(self, instrument: str) -> float:
        return self.iv_payload

    async def get_volatility_index_history(
        self,
        currency: str = "BTC",
        resolution: str = "1D",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()


class _ExistsButEmptyCache:
    def exists(self, exchange, data_type, instrument, start, end):
        return True

    def get(self, exchange, data_type, instrument, start, end):
        return None

    def put_range(self, *args, **kwargs):
        return None


@pytest.fixture
def stub_order_book() -> OrderBook:
    now = datetime.now(timezone.utc)
    return OrderBook(
        timestamp=now,
        instrument="BTC-PERPETUAL",
        bids=[OrderBookLevel(price=100.0, size=2.0)],
        asks=[OrderBookLevel(price=101.0, size=1.5)],
    )


@pytest.mark.asyncio
async def test_download_order_book_snapshots_falls_back_when_cache_exists_but_read_is_empty(
    stub_order_book: OrderBook,
) -> None:
    downloader = DeribitDataDownloader(cache=_ExistsButEmptyCache())
    downloader.client = _StubClient(stub_order_book, {"delta": 0.5}, 0.62)

    ts = [datetime.now(timezone.utc)]
    df = await downloader.download_order_book_snapshots("BTC-PERPETUAL", ts, use_cache=True)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert {"best_bid", "best_ask", "spread"}.issubset(df.columns)


@pytest.mark.asyncio
async def test_download_greeks_uses_datacache_api_without_type_errors(
    tmp_path,
    stub_order_book: OrderBook,
) -> None:
    cache = DataCache(base_dir=tmp_path)
    downloader = DeribitDataDownloader(cache=cache)
    downloader.client = _StubClient(
        stub_order_book, {"delta": 0.5, "gamma": 0.1, "theta": -0.2, "vega": 0.3, "rho": 0.01}, 0.62
    )

    result = await downloader.download_greeks("BTC-27DEC24-80000-C", use_cache=True)

    assert result is not None
    assert len(result) == 1
    assert result["delta"].iloc[0] == 0.5


@pytest.mark.asyncio
async def test_download_iv_uses_datacache_api_without_type_errors(
    tmp_path,
    stub_order_book: OrderBook,
) -> None:
    cache = DataCache(base_dir=tmp_path)
    downloader = DeribitDataDownloader(cache=cache)
    downloader.client = _StubClient(stub_order_book, {"delta": 0.5}, 0.72)

    first = await downloader.download_iv("BTC-27DEC24-80000-C", use_cache=True)
    second = await downloader.download_iv("BTC-27DEC24-80000-C", use_cache=True)

    assert first == pytest.approx(0.72)
    assert second == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_download_volatility_index_uses_cache_when_available(
    tmp_path, stub_order_book: OrderBook
) -> None:
    cache = DataCache(base_dir=tmp_path)
    downloader = DeribitDataDownloader(cache=cache)
    downloader.client = _StubClient(stub_order_book, {"delta": 0.5}, 0.5)

    now = datetime.now(timezone.utc)
    expected = pd.DataFrame(
        {
            "timestamp": [now - timedelta(days=1), now],
            "open": [40.0, 41.0],
            "high": [41.0, 42.0],
            "low": [39.5, 40.5],
            "close": [40.5, 41.5],
        }
    )
    cache.put_range(expected, "deribit", "dvol", "BTC")

    cached = await downloader.download_volatility_index("BTC", period_days=1, use_cache=True)

    assert not cached.empty
    assert set(expected.columns).issubset(cached.columns)
