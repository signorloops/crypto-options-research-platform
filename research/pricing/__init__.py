"""
期权定价模型 (币本位 / Coin-margined)。

支持:
- 币本位期权定价 (Inverse Option Model)
- 币本位希腊字母计算
- Put-Call Parity (币本位版本)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "InverseOptionPricer",
    "inverse_option_parity",
    "CryptoOptionModelZoo",
    "OptionQuote",
    "QuantoInverseOptionPricer",
    "QuantoInverseGreeks",
    "RoughVolConfig",
    "RoughVolatilityPricer",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "InverseOptionPricer": ("research.pricing.inverse_options", "InverseOptionPricer"),
    "inverse_option_parity": ("research.pricing.inverse_options", "inverse_option_parity"),
    "CryptoOptionModelZoo": ("research.pricing.model_zoo", "CryptoOptionModelZoo"),
    "OptionQuote": ("research.pricing.model_zoo", "OptionQuote"),
    "QuantoInverseOptionPricer": ("research.pricing.quanto_inverse", "QuantoInverseOptionPricer"),
    "QuantoInverseGreeks": ("research.pricing.quanto_inverse", "QuantoInverseGreeks"),
    "RoughVolConfig": ("research.pricing.rough_volatility", "RoughVolConfig"),
    "RoughVolatilityPricer": ("research.pricing.rough_volatility", "RoughVolatilityPricer"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'research.pricing' has no attribute '{name}'")

    module_name, symbol_name = target
    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
