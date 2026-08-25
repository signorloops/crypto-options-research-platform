"""Unit tests for Deribit client behavior without network calls."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core.types import OptionType
from data.downloaders.deribit import DeribitClient


@pytest.mark.asyncio
async def test_get_instruments_parses_option_contracts(monkeypatch):
    client = DeribitClient()
    payload = [
        {
            "base_currency": "BTC",
            "strike": 50000,
            "expiration_timestamp": int(
                datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
            ),
            "option_type": "call",
        },
        {
            "base_currency": "BTC",
            "strike": 30000,
            "expiration_timestamp": int(
                datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
            ),
            "option_type": "put",
        },
    ]
    monkeypatch.setattr(client, "_request", AsyncMock(return_value=payload))

    contracts = await client.get_instruments(currency="BTC", instrument_type="option")

    assert len(contracts) == 2
    assert contracts[0].option_type == OptionType.CALL
    assert contracts[1].option_type == OptionType.PUT
    assert contracts[0].expiry.tzinfo == timezone.utc
    assert contracts[1].expiry.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_get_order_book_and_tick_delegate_to_request(monkeypatch):
    client = DeribitClient()
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(
            return_value={
                "bids": [[100.0, 2.0], [99.0, 1.0]],
                "asks": [[101.0, 1.5], [102.0, 0.5]],
            }
        ),
    )

    order_book = await client.get_order_book("BTC-PERPETUAL", depth=2)
    tick = await client.get_tick("BTC-PERPETUAL")

    assert order_book.best_bid == 100.0
    assert order_book.best_ask == 101.0
    assert tick.bid == 100.0
    assert tick.ask == 101.0


@pytest.mark.asyncio
async def test_get_option_greeks_and_iv_from_ticker(monkeypatch):
    client = DeribitClient()
    monkeypatch.setattr(
        client,
        "get_ticker",
        AsyncMock(
            return_value={
                "mark_iv": 0.56,
                "greeks": {"delta": 0.2, "gamma": 0.01, "theta": -0.02, "vega": 0.3, "rho": 0.05},
            }
        ),
    )

    greeks = await client.get_option_greeks("BTC-TEST-C")
    iv = await client.get_option_iv("BTC-TEST-C")

    assert greeks == {"delta": 0.2, "gamma": 0.01, "theta": -0.02, "vega": 0.3, "rho": 0.05}
    assert iv == pytest.approx(0.56)


@pytest.mark.asyncio
async def test_get_volatility_index_history_transforms_percentages(monkeypatch):
    client = DeribitClient()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(return_value={"data": [[now_ms, 50, 60, 40, 55]]}),
    )

    result = await client.get_volatility_index_history("BTC")

    assert list(result.columns) == ["timestamp", "open", "high", "low", "close"]
    assert result["open"].iloc[0] == pytest.approx(0.5)
    assert result["close"].iloc[0] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_get_trades_paginates_and_applies_rate_limit(monkeypatch):
    client = DeribitClient()
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    end = datetime.now(timezone.utc)
    first_ts = int((start + timedelta(minutes=1)).timestamp() * 1000)

    responses = [
        {
            "trades": [
                {
                    "timestamp": first_ts,
                    "price": 100.0,
                    "amount": 0.2,
                    "direction": "buy",
                    "trade_id": "1",
                }
            ]
        },
        {"trades": []},
    ]

    async def _mock_request(method, params):
        return responses.pop(0)

    sleep_mock = AsyncMock()
    monkeypatch.setattr(client, "_request", _mock_request)
    monkeypatch.setattr("data.downloaders.deribit.asyncio.sleep", sleep_mock)

    trades = await client.get_trades("BTC-PERPETUAL", start, end, limit=5)

    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert trades[0].timestamp.tzinfo == timezone.utc
    sleep_mock.assert_awaited()


def test_trade_query_params_request_ascending_sort():
    """Cursor advancement assumes ascending pages, so sorting must be explicit.

    Without sorting="asc" the API may return newest-first, and batch[-1]
    would move the cursor backwards through the window.
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    params = DeribitClient._build_trade_query_params(
        instrument="BTC-PERPETUAL",
        current_start=start,
        end=start + timedelta(hours=2),
        limit=1000,
    )
    assert params["sorting"] == "asc"


def test_advance_trade_cursor_is_monotonic_for_unsorted_page():
    """A newest-first page must not move the cursor backwards."""
    start_ms = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    batch = [
        {"timestamp": start_ms + 2000},
        {"timestamp": start_ms + 1000},
    ]
    cursor = DeribitClient._advance_trade_cursor(batch)
    assert cursor == datetime.fromtimestamp((start_ms + 2000) / 1000, tz=timezone.utc)


def test_dedupe_trades_keyed_by_trade_id():
    """Boundary-millisecond trades are re-fetched across pages; dedupe them.

    The old cursor used +1ms, which silently dropped trades sharing the
    boundary millisecond.
    """
    trades = [
        _trade("1", 1000),
        _trade("1", 1000),  # duplicate from page overlap
        _trade("2", 2000),
        _trade(None, 3000),
        _trade(None, 3000),  # duplicate without a trade_id
    ]
    deduped = DeribitClient._dedupe_trades(trades)
    assert [t.trade_id for t in deduped] == ["1", "2", None]
    assert len(deduped) == 3


def _trade(trade_id, ts_ms):
    from core.types import OrderSide, Trade

    return Trade(
        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
        instrument="BTC-PERPETUAL",
        price=100.0,
        size=0.2,
        side=OrderSide.BUY,
        trade_id=trade_id,
    )


@pytest.mark.asyncio
async def test_get_trades_dedupes_boundary_overlap_and_never_stalls(monkeypatch):
    """get_trades keeps boundary trades and always makes forward progress.

    Page 1 ends on trades T1/T2 sharing the boundary millisecond; page 2
    re-returns T2 (overlap) plus T3. The result must contain T1, T2, T3
    exactly once, and the pagination must terminate.
    """
    client = DeribitClient()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    boundary_ms = int((start + timedelta(minutes=1)).timestamp() * 1000)

    def page(trade_id, ts, price):
        return {
            "timestamp": ts,
            "price": price,
            "amount": 0.2,
            "direction": "buy",
            "trade_id": trade_id,
        }

    responses = [
        {"trades": [page("T1", boundary_ms, 100.0), page("T2", boundary_ms, 101.0)]},
        # Overlapping page: T2 again, then a strictly later trade.
        {"trades": [page("T2", boundary_ms, 101.0), page("T3", boundary_ms + 1500, 102.0)]},
        {"trades": []},
    ]

    async def _mock_request(method, params):
        assert responses, "pagination did not terminate"
        return responses.pop(0)

    monkeypatch.setattr(client, "_request", _mock_request)
    monkeypatch.setattr(
        "data.downloaders.deribit.asyncio.sleep", AsyncMock()
    )

    trades = await client.get_trades("BTC-PERPETUAL", start, end, limit=5)

    assert [t.trade_id for t in trades] == ["T1", "T2", "T3"]


@pytest.mark.asyncio
async def test_get_historical_volatility_uses_latest_value(monkeypatch):
    client = DeribitClient()
    monkeypatch.setattr(
        client,
        "_request",
        AsyncMock(return_value=[[1, 35.0], [2, 40.0]]),
    )

    value = await client.get_historical_volatility("BTC")
    assert value == pytest.approx(0.4)
