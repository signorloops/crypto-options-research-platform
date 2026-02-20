"""
Tests for data validation module.
"""
from datetime import datetime, timedelta, timezone

import pytest

from core.validation import (
    BacktestConfig,
    DataValidationError,
    DownloadRequest,
    GreeksData,
    OptionContractData,
    OrderBookData,
    OrderBookLevelData,
    TickData,
    TradeData,
    WebSocketConfig,
    validate_datetime_range,
    validate_instrument_name,
    validate_positive,
    validate_price,
)


class TestTickData:
    """Test TickData validation."""

    def test_valid_tick(self):
        """Test valid tick data."""
        tick = TickData(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-USD",
            bid=50000.0,
            ask=50001.0,
            bid_size=1.5,
            ask_size=2.0
        )
        assert tick.bid == 50000.0
        assert tick.ask == 50001.0

    def test_invalid_bid_negative(self):
        """Test bid cannot be negative."""
        with pytest.raises(Exception):  # pydantic ValidationError
            TickData(
                timestamp=datetime.now(timezone.utc),
                instrument="BTC-USD",
                bid=-1.0,
                ask=50001.0,
                bid_size=1.5,
                ask_size=2.0
            )

    def test_invalid_ask_less_than_bid(self):
        """Test ask must be >= bid."""
        with pytest.raises(Exception) as exc_info:
            TickData(
                timestamp=datetime.now(timezone.utc),
                instrument="BTC-USD",
                bid=50000.0,
                ask=49999.0,
                bid_size=1.5,
                ask_size=2.0
            )
        assert "ask" in str(exc_info.value).lower()

    def test_invalid_instrument_empty(self):
        """Test instrument cannot be empty."""
        with pytest.raises(Exception):
            TickData(
                timestamp=datetime.now(timezone.utc),
                instrument="",
                bid=50000.0,
                ask=50001.0,
                bid_size=1.5,
                ask_size=2.0
            )


class TestTradeData:
    """Test TradeData validation."""

    def test_valid_trade(self):
        """Test valid trade data."""
        trade = TradeData(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-USD",
            price=50000.0,
            size=1.0,
            side="BUY"
        )
        assert trade.price == 50000.0
        assert trade.side.value == "BUY"

    def test_invalid_price_zero(self):
        """Test price must be positive."""
        with pytest.raises(Exception):
            TradeData(
                timestamp=datetime.now(timezone.utc),
                instrument="BTC-USD",
                price=0.0,
                size=1.0,
                side="BUY"
            )

    def test_invalid_size_negative(self):
        """Test size cannot be negative."""
        with pytest.raises(Exception):
            TradeData(
                timestamp=datetime.now(timezone.utc),
                instrument="BTC-USD",
                price=50000.0,
                size=-1.0,
                side="BUY"
            )


class TestOrderBookData:
    """Test OrderBookData validation."""

    def test_valid_orderbook(self):
        """Test valid order book."""
        ob = OrderBookData(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-USD",
            bids=[
                OrderBookLevelData(price=50000.0, size=1.0),
                OrderBookLevelData(price=49999.0, size=2.0),
            ],
            asks=[
                OrderBookLevelData(price=50001.0, size=1.5),
                OrderBookLevelData(price=50002.0, size=2.5),
            ]
        )
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2

    def test_invalid_spread(self):
        """Test best ask must be > best bid."""
        with pytest.raises(Exception) as exc_info:
            OrderBookData(
                timestamp=datetime.now(timezone.utc),
                instrument="BTC-USD",
                bids=[OrderBookLevelData(price=50000.0, size=1.0)],
                asks=[OrderBookLevelData(price=49999.0, size=1.0)],
            )
        assert "spread" in str(exc_info.value).lower() or "ask" in str(exc_info.value).lower()

    def test_invalid_price_zero(self):
        """Test order book level price must be positive."""
        with pytest.raises(Exception):
            OrderBookLevelData(price=0.0, size=1.0)


class TestOptionContractData:
    """Test OptionContractData validation."""

    def test_valid_option(self):
        """Test valid option contract."""
        opt = OptionContractData(
            underlying="BTC",
            strike=50000.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            option_type="CALL"
        )
        assert opt.underlying == "BTC"
        assert opt.option_type.value == "CALL"

    def test_underlying_uppercase(self):
        """Test underlying is converted to uppercase."""
        opt = OptionContractData(
            underlying="btc",
            strike=50000.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            option_type="CALL"
        )
        assert opt.underlying == "BTC"

    def test_invalid_strike_zero(self):
        """Test strike must be positive."""
        with pytest.raises(Exception):
            OptionContractData(
                underlying="BTC",
                strike=0.0,
                expiry=datetime.now(timezone.utc) + timedelta(days=30),
                option_type="CALL"
            )


class TestGreeksData:
    """Test GreeksData validation."""

    def test_valid_greeks(self):
        """Test valid Greeks data."""
        greeks = GreeksData(
            delta=0.5,
            gamma=0.01,
            theta=-0.1,
            vega=0.2,
            rho=0.05
        )
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.01

    def test_invalid_delta_out_of_range(self):
        """Test delta must be between -1 and 1."""
        with pytest.raises(Exception):
            GreeksData(
                delta=1.5,
                gamma=0.01,
                theta=-0.1,
                vega=0.2,
                rho=0.05
            )

    def test_invalid_gamma_negative(self):
        """Test gamma cannot be negative."""
        with pytest.raises(Exception):
            GreeksData(
                delta=0.5,
                gamma=-0.01,
                theta=-0.1,
                vega=0.2,
                rho=0.05
            )

    def test_valid_iv_range(self):
        """Test implied volatility range."""
        greeks = GreeksData(
            delta=0.5,
            gamma=0.01,
            theta=-0.1,
            vega=0.2,
            rho=0.05,
            iv=0.5
        )
        assert greeks.iv == 0.5

    def test_invalid_iv_too_high(self):
        """Test IV cannot exceed 500%."""
        with pytest.raises(Exception):
            GreeksData(
                delta=0.5,
                gamma=0.01,
                theta=-0.1,
                vega=0.2,
                rho=0.05,
                iv=10.0
            )


class TestBacktestConfig:
    """Test BacktestConfig validation."""

    def test_valid_config(self):
        """Test valid backtest configuration."""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            instruments=["BTC-USD", "ETH-USD"],
            initial_capital=100000.0,
            commission_rate=0.001,
        )
        assert len(config.instruments) == 2

    def test_invalid_date_range(self):
        """Test end_date must be after start_date."""
        with pytest.raises(Exception) as exc_info:
            BacktestConfig(
                start_date=datetime(2024, 6, 1),
                end_date=datetime(2024, 1, 1),
                instruments=["BTC-USD"],
            )
        assert "end_date" in str(exc_info.value).lower()

    def test_invalid_empty_instruments(self):
        """Test instruments cannot be empty."""
        with pytest.raises(Exception):
            BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 6, 1),
                instruments=[],
            )

    def test_invalid_duplicate_instruments(self):
        """Test instruments must be unique."""
        with pytest.raises(Exception) as exc_info:
            BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 6, 1),
                instruments=["BTC-USD", "BTC-USD"],
            )
        assert "unique" in str(exc_info.value).lower()

    def test_invalid_commission_rate(self):
        """Test commission rate must be reasonable."""
        with pytest.raises(Exception):
            BacktestConfig(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 6, 1),
                instruments=["BTC-USD"],
                commission_rate=0.5,  # 50% is too high
            )


class TestDownloadRequest:
    """Test DownloadRequest validation."""

    def test_valid_request(self):
        """Test valid download request."""
        req = DownloadRequest(
            exchange="deribit",
            data_type="trades",
            instrument="BTC-27DEC24-80000-C",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
        )
        assert req.exchange == "deribit"

    def test_invalid_exchange(self):
        """Test exchange must be valid."""
        with pytest.raises(Exception):
            DownloadRequest(
                exchange="invalid_exchange",
                data_type="trades",
                instrument="BTC",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31),
            )

    def test_invalid_date_range_too_long(self):
        """Test date range cannot exceed 1 year."""
        with pytest.raises(Exception) as exc_info:
            DownloadRequest(
                exchange="deribit",
                data_type="trades",
                instrument="BTC",
                start=datetime(2023, 1, 1),
                end=datetime(2024, 6, 1),  # > 1 year
            )
        assert "year" in str(exc_info.value).lower() or "range" in str(exc_info.value).lower()


class TestWebSocketConfig:
    """Test WebSocketConfig validation."""

    def test_valid_config(self):
        """Test valid WebSocket configuration."""
        config = WebSocketConfig(
            reconnect_interval=5.0,
            max_reconnects=10,
            ping_interval=20.0,
            ping_timeout=10.0,
        )
        assert config.reconnect_interval == 5.0

    def test_invalid_ping_timeout(self):
        """Test ping_timeout must be less than ping_interval."""
        with pytest.raises(Exception) as exc_info:
            WebSocketConfig(
                ping_interval=10.0,
                ping_timeout=15.0,  # > interval
            )
        assert "timeout" in str(exc_info.value).lower()

    def test_default_values(self):
        """Test default values are set correctly."""
        config = WebSocketConfig()
        assert config.reconnect_interval == 5.0
        assert config.max_reconnects == 10


class TestCustomValidators:
    """Test custom validator functions."""

    def test_validate_price_valid(self):
        """Test valid price validation."""
        assert validate_price(100.0) == 100.0

    def test_validate_price_negative(self):
        """Test price cannot be negative."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_price(-1.0)
        assert exc_info.value.field == "price"

    def test_validate_price_zero(self):
        """Test price cannot be zero (by default)."""
        with pytest.raises(DataValidationError):
            validate_price(0.0)

    def test_validate_price_zero_allowed(self):
        """Test price can be zero when allowed."""
        assert validate_price(0.0, allow_zero=True) == 0.0

    def test_validate_price_none(self):
        """Test price cannot be None."""
        with pytest.raises(DataValidationError):
            validate_price(None)

    def test_validate_positive_valid(self):
        """Test valid positive validation."""
        assert validate_positive(100.0) == 100.0

    def test_validate_positive_zero_strict(self):
        """Test zero fails in strict mode."""
        with pytest.raises(DataValidationError):
            validate_positive(0.0, strict=True)

    def test_validate_positive_zero_non_strict(self):
        """Test zero passes in non-strict mode."""
        assert validate_positive(0.0, strict=False) == 0.0

    def test_validate_instrument_name_valid(self):
        """Test valid instrument name."""
        assert validate_instrument_name("btc-usd") == "BTC-USD"

    def test_validate_instrument_name_empty(self):
        """Test empty instrument name fails."""
        with pytest.raises(DataValidationError):
            validate_instrument_name("")

    def test_validate_instrument_name_invalid_chars(self):
        """Test instrument name with invalid characters fails."""
        with pytest.raises(DataValidationError):
            validate_instrument_name("BTC@USD")

    def test_validate_datetime_range_valid(self):
        """Test valid datetime range."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        result = validate_datetime_range(start, end)
        assert result == (start, end)

    def test_validate_datetime_range_invalid(self):
        """Test invalid datetime range."""
        start = datetime(2024, 6, 1)
        end = datetime(2024, 1, 1)
        with pytest.raises(DataValidationError):
            validate_datetime_range(start, end)

    def test_validate_datetime_range_too_long(self):
        """Test date range exceeding maximum."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, tzinfo=timezone.utc)  # > 365 days
        with pytest.raises(DataValidationError):
            validate_datetime_range(start, end, max_range_days=365)

    def test_validate_datetime_range_future_not_allowed(self):
        """Test future dates not allowed by default."""
        start = datetime.now(timezone.utc) + timedelta(days=1)
        end = datetime.now(timezone.utc) + timedelta(days=2)
        with pytest.raises(DataValidationError):
            validate_datetime_range(start, end)

    def test_validate_datetime_range_future_allowed(self):
        """Test future dates allowed when specified."""
        start = datetime.now(timezone.utc) + timedelta(days=1)
        end = datetime.now(timezone.utc) + timedelta(days=2)
        result = validate_datetime_range(start, end, allow_future=True)
        assert result == (start, end)

    def test_data_validation_error_with_field(self):
        """Test DataValidationError with field info."""
        error = DataValidationError("Invalid value", field="price", value=-1)
        assert "price" in str(error)
        assert "-1" in str(error)

    def test_data_validation_error_without_field(self):
        """Test DataValidationError without field info."""
        error = DataValidationError("General error")
        assert "General error" in str(error)
