"""Unit tests for OKX client logic without external API calls."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from core.types import OptionType, OrderSide
from data.downloaders.okx import OKXAPIError, OKXClient


@pytest.mark.asyncio
async def test_get_instruments_filters_live_only(monkeypatch):
    client = OKXClient()
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(
            return_value={
                "data": [
                    {"instId": "BTC-USD-SWAP", "state": "live"},
                    {"instId": "ETH-USD-SWAP", "state": "suspend"},
                ]
            }
        ),
    )

    result = await client.get_instruments(currency="BTC", instrument_type="SWAP")
    assert result == ["BTC-USD-SWAP"]


@pytest.mark.asyncio
async def test_get_option_instruments_validates_underlying():
    client = OKXClient()
    with pytest.raises(ValueError):
        await client.get_option_instruments("SOL-USD")


@pytest.mark.asyncio
async def test_get_option_instruments_parses_valid_rows(monkeypatch):
    client = OKXClient()
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(
            return_value={
                "data": [
                    {
                        "instId": "BTC-USD-240628-50000-C",
                        "state": "live",
                        "lotSz": "0.1",
                        "tickSz": "0.01",
                    },
                    {"instId": "BTC-USD-BAD", "state": "live"},
                ]
            }
        ),
    )

    contracts = await client.get_option_instruments("BTC-USD")
    assert len(contracts) == 1
    assert contracts[0].option_type == OptionType.CALL
    assert contracts[0].underlying == "BTC-USD"


@pytest.mark.asyncio
async def test_market_data_accessors_parse_payload(monkeypatch):
    client = OKXClient()

    async def _mock_request(endpoint, params=None, timeout=30.0):
        if endpoint.endswith("/books"):
            return {"data": [{"bids": [["100", "2"]], "asks": [["101", "1.5"]]}]}
        if endpoint.endswith("/ticker"):
            return {"data": [{"bidPx": "100", "askPx": "101", "bidSz": "1.2", "askSz": "0.9"}]}
        if endpoint.endswith("/index-tickers"):
            return {"data": [{"idxPx": "50000"}]}
        raise AssertionError(f"unexpected endpoint: {endpoint}")

    monkeypatch.setattr(client, "_request", _mock_request)

    order_book = await client.get_order_book("BTC-USD-SWAP")
    ticker = await client.get_ticker("BTC-USD-SWAP")
    index_price = await client.get_index_price("BTC-USD")

    assert order_book.best_bid == 100.0
    assert order_book.best_ask == 101.0
    assert ticker.bid == 100.0
    assert ticker.ask == 101.0
    assert index_price == pytest.approx(50000.0)


@pytest.mark.asyncio
async def test_get_klines_parses_history_and_sorts(monkeypatch):
    client = OKXClient()
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(
            return_value={
                "data": [
                    [str(now), "1", "2", "0.5", "1.5", "100", "0"],
                    [str(now - 60_000), "2", "3", "1", "2.5", "120", "0"],
                ]
            }
        ),
    )

    df = await client.get_klines("BTC-USD", interval="1m", limit=2)

    assert not df.empty
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "vol_ccy"]
    assert df["timestamp"].is_monotonic_increasing


@pytest.mark.asyncio
async def test_get_trades_maps_side_and_fields(monkeypatch):
    client = OKXClient()
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(
            return_value={
                "data": [
                    {
                        "ts": "1700000000000",
                        "px": "100",
                        "sz": "0.2",
                        "side": "buy",
                        "tradeId": "a",
                    },
                    {
                        "ts": "1700000001000",
                        "px": "101",
                        "sz": "0.3",
                        "side": "sell",
                        "tradeId": "b",
                    },
                ]
            }
        ),
    )

    start = datetime.now(timezone.utc) - timedelta(minutes=10)
    end = datetime.now(timezone.utc)
    trades = await client.get_trades("BTC-USD-SWAP", start, end)

    assert len(trades) == 2
    assert trades[0].side == OrderSide.BUY
    assert trades[1].side == OrderSide.SELL
    assert trades[0].timestamp.tzinfo == timezone.utc
    assert trades[1].timestamp.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_option_volatility_history_error_returns_empty(monkeypatch):
    client = OKXClient()
    monkeypatch.setattr(client, "_request", AsyncMock(side_effect=OKXAPIError("bad")))

    result = await client.get_option_volatility_history("BTC-USD", period_days=7)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


@pytest.mark.asyncio
async def test_current_iv_term_structure_extracts_atm(monkeypatch):
    client = OKXClient()
    expiry = int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp() * 1000)
    monkeypatch.setattr(
        client,
        "get_option_market_data",
        AsyncMock(
            return_value=[
                {"expTime": str(expiry), "stk": "50000", "markVol": "0.5", "ulyPx": "50020"},
                {"expTime": str(expiry), "stk": "70000", "markVol": "0.9", "ulyPx": "50020"},
            ]
        ),
    )

    df = await client.get_current_iv_term_structure("BTC-USD")
    assert not df.empty
    assert "atm_iv" in df.columns
    assert df["atm_iv"].iloc[0] == pytest.approx(0.5)
