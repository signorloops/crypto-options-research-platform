"""Tests for lazy package exports in top-level research/strategy packages."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "symbol_name"),
    [
        ("data", "DataCache"),
        ("research.pricing", "InverseOptionPricer"),
        ("research.volatility", "realized_volatility"),
        ("research.hedging", "AdaptiveDeltaHedger"),
        ("research.signals", "BOCDConfig"),
        ("strategies.arbitrage", "BasisArbitrage"),
    ],
)
def test_lazy_exports_resolve_known_symbols(module_name: str, symbol_name: str) -> None:
    module = importlib.import_module(module_name)
    assert symbol_name in module.__all__
    resolved = getattr(module, symbol_name)
    assert resolved is not None


@pytest.mark.parametrize(
    ("module_name", "missing_symbol"),
    [
        ("data", "__missing_symbol__"),
        ("research.pricing", "__missing_symbol__"),
        ("research.volatility", "__missing_symbol__"),
        ("research.hedging", "__missing_symbol__"),
        ("research.signals", "__missing_symbol__"),
        ("strategies.arbitrage", "__missing_symbol__"),
    ],
)
def test_lazy_exports_raise_for_unknown_symbols(module_name: str, missing_symbol: str) -> None:
    module = importlib.import_module(module_name)
    with pytest.raises(AttributeError):
        getattr(module, missing_symbol)
