"""
Custom validators and validation utilities.
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import numpy as np

from core.exceptions import ValidationError

# Alias for backward compatibility
DataValidationError = ValidationError


def _require_numeric_value(
    value: Any,
    *,
    field_name: str,
    none_message: str,
    type_message: str,
    finite_message: str = "Value must be finite",
) -> float:
    if value is None:
        raise DataValidationError(none_message, field=field_name, value=value)
    if not isinstance(value, (int, float)):
        raise DataValidationError(type_message, field=field_name, value=value)
    if not np.isfinite(value):
        raise DataValidationError(finite_message, field=field_name, value=value)
    return float(value)


def validate_price(price: float, field_name: str = "price", allow_zero: bool = False) -> float:
    """
    Validate price value.

    Args:
        price: Price value to validate
        field_name: Name of the field for error messages
        allow_zero: Whether to allow zero prices

    Returns:
        Validated price

    Raises:
        DataValidationError: If price is invalid
    """
    numeric_price = _require_numeric_value(
        price,
        field_name=field_name,
        none_message="Price cannot be None",
        type_message="Price must be a number",
        finite_message="Price must be finite",
    )
    if numeric_price < 0:
        raise DataValidationError("Price cannot be negative", field=field_name, value=price)
    if not allow_zero and numeric_price == 0:
        raise DataValidationError("Price cannot be zero", field=field_name, value=price)
    return numeric_price


def validate_positive(value: float, field_name: str = "value", strict: bool = True) -> float:
    """
    Validate that value is positive.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        strict: If True, value must be > 0; if False, >= 0

    Returns:
        Validated value

    Raises:
        DataValidationError: If value is invalid
    """
    numeric = _require_numeric_value(
        value,
        field_name=field_name,
        none_message="Value cannot be None",
        type_message="Value must be a number",
    )
    if strict and numeric <= 0:
        raise DataValidationError("Value must be positive", field=field_name, value=value)
    if not strict and numeric < 0:
        raise DataValidationError("Value cannot be negative", field=field_name, value=value)
    return numeric


def validate_instrument_name(name: str, field_name: str = "instrument") -> str:
    """
    Validate instrument/trading pair name.

    Args:
        name: Instrument name to validate
        field_name: Name of the field for error messages

    Returns:
        Validated instrument name (uppercase)

    Raises:
        DataValidationError: If name is invalid
    """
    if not name:
        raise DataValidationError("Instrument name cannot be empty", field=field_name, value=name)
    if not isinstance(name, str):
        raise DataValidationError("Instrument name must be a string", field=field_name, value=name)
    normalized_name = name.strip().upper()
    _validate_instrument_name_length(normalized_name, field_name)
    _validate_instrument_name_chars(normalized_name, field_name)
    return normalized_name


def _validate_instrument_name_length(name: str, field_name: str) -> None:
    if len(name) < 1 or len(name) > 50:
        raise DataValidationError(
            "Instrument name length must be between 1 and 50",
            field=field_name,
            value=name,
        )


def _validate_instrument_name_chars(name: str, field_name: str) -> None:
    if not re.match(r"^[A-Z0-9\-_/]+$", name):
        raise DataValidationError(
            "Instrument name contains invalid characters",
            field=field_name,
            value=name,
        )


def validate_datetime_range(
    start: datetime, end: datetime, max_range_days: Optional[int] = None, allow_future: bool = False
) -> Tuple[datetime, datetime]:
    """Validate datetime range and return normalized boundaries."""
    _validate_datetime_inputs(start=start, end=end)
    _validate_datetime_order(start=start, end=end)
    if not allow_future:
        _validate_not_future(start=start, end=end)
    _validate_datetime_max_range(start=start, end=end, max_range_days=max_range_days)
    return start, end


def _validate_datetime_inputs(*, start: datetime, end: datetime) -> None:
    if start is None or end is None:
        raise DataValidationError(
            "Start and end dates cannot be None", field="datetime_range", value=None
        )
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise DataValidationError(
            "Start and end must be datetime objects", field="datetime_range", value=(start, end)
        )


def _validate_datetime_order(*, start: datetime, end: datetime) -> None:
    if end <= start:
        raise DataValidationError(
            f"End date ({end}) must be after start date ({start})",
            field="datetime_range",
            value=(start, end),
        )


def _validate_datetime_max_range(
    *, start: datetime, end: datetime, max_range_days: Optional[int]
) -> None:
    if max_range_days is None:
        return
    range_days = (end - start).days
    if range_days > max_range_days:
        raise DataValidationError(
            f"Date range ({range_days} days) exceeds maximum ({max_range_days} days)",
            field="datetime_range",
            value=(start, end),
        )


def _validate_not_future(*, start: datetime, end: datetime) -> None:
    now = datetime.now(timezone.utc)
    if start > now:
        raise DataValidationError(
            f"Start date ({start}) is in the future", field="start", value=start
        )
    if end > now:
        raise DataValidationError(f"End date ({end}) is in the future", field="end", value=end)


def validate_exchange_name(exchange: str) -> str:
    """
    Validate exchange name.

    Args:
        exchange: Exchange name to validate

    Returns:
        Validated exchange name (lowercase)

    Raises:
        DataValidationError: If exchange is invalid
    """
    valid_exchanges = {"deribit", "binance", "okx", "bybit"}

    if not exchange:
        raise DataValidationError("Exchange name cannot be empty", field="exchange", value=exchange)

    if not isinstance(exchange, str):
        raise DataValidationError(
            "Exchange name must be a string", field="exchange", value=exchange
        )

    exchange = exchange.lower().strip()

    if exchange not in valid_exchanges:
        raise DataValidationError(
            f"Invalid exchange '{exchange}'. Must be one of: {valid_exchanges}",
            field="exchange",
            value=exchange,
        )

    return exchange


def _validate_order_book_level(level: Any, side: str, index: int) -> Tuple[float, float]:
    if not isinstance(level, (tuple, list)) or len(level) != 2:
        raise DataValidationError(
            f"{side.capitalize()} level must be a (price, size) pair",
            field=f"{side}[{index}]",
            value=level,
        )
    price, size = level
    if not isinstance(price, (int, float)) or not np.isfinite(price):
        raise DataValidationError(
            f"{side.capitalize()} price must be a finite number",
            field=f"{side}[{index}].price",
            value=price,
        )
    if not isinstance(size, (int, float)) or not np.isfinite(size):
        raise DataValidationError(
            f"{side.capitalize()} size must be a finite number",
            field=f"{side}[{index}].size",
            value=size,
        )
    return float(price), float(size)


def _validate_order_book_side(levels: list, side: str) -> None:
    for i, level in enumerate(levels):
        price, size = _validate_order_book_level(level, side, i)
        if price <= 0:
            raise DataValidationError(
                f"{side[:-1].capitalize()} price must be positive",
                field=f"{side}[{i}].price",
                value=price,
            )
        if size < 0:
            raise DataValidationError(
                f"{side[:-1].capitalize()} size cannot be negative",
                field=f"{side}[{i}].size",
                value=size,
            )


def _validate_order_book_not_empty(bids: list, asks: list) -> None:
    if not bids and not asks:
        raise DataValidationError(
            "Order book cannot be empty", field="order_book", value=(bids, asks)
        )


def _validate_order_book_spread(bids: list, asks: list) -> None:
    if not bids or not asks:
        return
    best_bid = max(b[0] for b in bids)
    best_ask = min(a[0] for a in asks)
    if best_ask <= best_bid:
        raise DataValidationError(
            f"Invalid spread: best_ask ({best_ask}) <= best_bid ({best_bid})",
            field="spread",
            value=(best_bid, best_ask),
        )


def validate_order_book(bids: list, asks: list) -> Tuple[list, list]:
    """
    Validate order book data.

    Args:
        bids: List of bid levels [(price, size), ...]
        asks: List of ask levels [(price, size), ...]

    Returns:
        Validated (bids, asks) tuple

    Raises:
        DataValidationError: If order book is invalid
    """
    _validate_order_book_not_empty(bids, asks)
    _validate_order_book_side(bids, "bids")
    _validate_order_book_side(asks, "asks")
    _validate_order_book_spread(bids, asks)
    return bids, asks


def _greek_validation_errors(delta: float, gamma: float, vega: float) -> list[str]:
    checks = [
        (-1 <= delta <= 1, f"Delta ({delta}) must be between -1 and 1"),
        (gamma >= 0, f"Gamma ({gamma}) must be non-negative"),
        (vega >= 0, f"Vega ({vega}) must be non-negative"),
    ]
    return [message for is_valid, message in checks if not is_valid]


def validate_greeks(delta: float, gamma: float, theta: float, vega: float) -> dict:
    """Validate Greeks values."""
    errors = _greek_validation_errors(delta, gamma, vega)
    if errors:
        raise DataValidationError(
            "; ".join(errors), field="greeks", value=(delta, gamma, theta, vega)
        )
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
