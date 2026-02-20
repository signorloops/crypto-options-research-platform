"""
Tests for OKX API client.
"""
from datetime import datetime, timedelta, timezone

import pytest

from data.downloaders.okx import OKXClient, OKXAPIError


class TestOKXClient:
    """Test OKX API client."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connection(self):
        """Test basic connection."""
        client = OKXClient()
        await client.connect()
        assert client._session is not None
        await client.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_instruments_spot(self):
        """Test fetching spot instruments."""
        client = OKXClient()
        async with client:
            symbols = await client.get_instruments(currency="BTC", instrument_type="SPOT")
            assert len(symbols) > 0
            assert any("BTC" in s for s in symbols)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_option_instruments(self):
        """Test fetching option instruments."""
        client = OKXClient()
        async with client:
            # Test coin-margined options (BTC-USD)
            contracts = await client.get_option_instruments(underlying="BTC-USD")
            assert len(contracts) > 0

            # Check that all are coin-margined
            for contract in contracts:
                assert contract.is_coin_margined is True
                assert contract.inverse is True
                assert "USD" in contract.underlying

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_order_book(self):
        """Test fetching order book."""
        client = OKXClient()
        async with client:
            ob = await client.get_order_book("BTC-USD", depth=5)
            assert ob.best_bid > 0
            assert ob.best_ask > ob.best_bid
            assert len(ob.bids) <= 5
            assert len(ob.asks) <= 5

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_ticker(self):
        """Test fetching ticker."""
        client = OKXClient()
        async with client:
            tick = await client.get_ticker("BTC-USD")
            assert tick.bid > 0
            assert tick.ask > 0
            assert tick.bid_size >= 0
            assert tick.ask_size >= 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_klines(self):
        """Test fetching OHLCV data (coin-margined index)."""
        client = OKXClient()
        async with client:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=1)
            df = await client.get_klines(
                "BTC-USD",  # Coin-margined underlying
                interval="1H",
                start=start,
                end=end
            )
            # Note: OKX may return empty data for some time ranges
            if len(df) > 0:
                assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_index_price(self):
        """Test fetching index price."""
        client = OKXClient()
        async with client:
            price = await client.get_index_price("BTC-USD")
            assert price > 0

    def test_valid_underlyings(self):
        """Test that valid coin-margined underlyings are defined."""
        client = OKXClient()
        assert "BTC-USD" in client.VALID_UNDERLYINGS
        assert "ETH-USD" in client.VALID_UNDERLYINGS
        assert "BTC-USDT" not in client.VALID_UNDERLYINGS


class TestOKXValidation:
    """Test OKX input validation."""

    def test_rejects_usdt_underlying(self):
        """Test that USDT underlyings are rejected."""
        client = OKXClient.__new__(OKXClient)
        client.VALID_UNDERLYINGS = {"BTC-USD", "ETH-USD"}

        with pytest.raises(ValueError, match="Only coin-margined options supported"):
            # Use asyncio.run to call the async method
            import asyncio
            asyncio.run(client.get_option_instruments(underlying="BTC-USDT"))

    def test_rejects_usdc_underlying(self):
        """Test that USDC underlyings are rejected."""
        client = OKXClient.__new__(OKXClient)
        client.VALID_UNDERLYINGS = {"BTC-USD", "ETH-USD"}

        with pytest.raises(ValueError, match="Only coin-margined options supported"):
            import asyncio
            asyncio.run(client.get_option_instruments(underlying="BTC-USDC"))

    def test_rejects_invalid_underlying(self):
        """Test that invalid underlyings are rejected."""
        client = OKXClient.__new__(OKXClient)
        client.VALID_UNDERLYINGS = {"BTC-USD", "ETH-USD"}

        with pytest.raises(ValueError, match="Only coin-margined options supported"):
            import asyncio
            asyncio.run(client.get_option_instruments(underlying="BTC-USD-SWAP"))
