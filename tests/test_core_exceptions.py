"""Tests for custom exception hierarchy in core.exceptions."""

from core.exceptions import (
    APIError,
    CORPError,
    ConfigurationError,
    DataError,
    PricingError,
    StrategyError,
    ValidationError,
)


def test_corp_error_string_with_and_without_code():
    err_with_code = CORPError("boom", code="E1")
    err_without_code = CORPError("boom")

    assert str(err_with_code) == "[E1] boom"
    assert str(err_without_code) == "boom"


def test_validation_error_formats_field_and_value():
    err = ValidationError("invalid", field="price", value=-1, code="V1")
    fallback = ValidationError("invalid")

    assert err.field == "price"
    assert err.value == -1
    assert str(err) == "[V1] price=-1: invalid"
    assert str(fallback) == "[VALIDATION_ERROR] invalid"


def test_specialized_errors_store_extra_metadata():
    pricing = PricingError("pricing failed")
    api = APIError("api failed", status_code=503)
    data = DataError("data failed", source="okx")
    config = ConfigurationError("config failed")
    strategy = StrategyError("strategy failed", strategy_name="mm")

    assert pricing.code == "PRICING_ERROR"
    assert api.code == "API_ERROR" and api.status_code == 503
    assert data.code == "DATA_ERROR" and data.source == "okx"
    assert config.code == "CONFIG_ERROR"
    assert strategy.code == "STRATEGY_ERROR" and strategy.strategy_name == "mm"
