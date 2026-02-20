from __future__ import annotations

import math

import pytest

from core.validation.validators import (
    DataValidationError,
    validate_exchange_name,
    validate_greeks,
    validate_order_book,
    validate_positive,
)


def test_validate_exchange_name_rejects_non_string_input() -> None:
    with pytest.raises(DataValidationError):
        validate_exchange_name(123)  # type: ignore[arg-type]


def test_validate_positive_rejects_non_finite_values() -> None:
    with pytest.raises(DataValidationError):
        validate_positive(math.inf)


def test_validate_order_book_rejects_malformed_levels_with_validation_error() -> None:
    with pytest.raises(DataValidationError):
        validate_order_book([(100.0,)], [(101.0, 1.0)])


def test_validate_greeks_reports_all_invalid_components() -> None:
    with pytest.raises(DataValidationError) as exc:
        validate_greeks(delta=2.0, gamma=-0.1, theta=0.0, vega=-0.2)

    message = str(exc.value)
    assert "Delta" in message
    assert "Gamma" in message
    assert "Vega" in message
