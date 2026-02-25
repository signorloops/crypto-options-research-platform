"""
套利策略模块。

包含:
- 跨交易所套利 (Cross-exchange arbitrage)
- 期现套利 (Basis trading / Cash-and-carry)
- 期权盒式套利 (Option box spread)
- 转换套利 (Conversion/Reversal arbitrage)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CrossExchangeArbitrage",
    "BasisArbitrage",
    "OptionBoxArbitrage",
    "ConversionArbitrage",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "CrossExchangeArbitrage": ("strategies.arbitrage.cross_exchange", "CrossExchangeArbitrage"),
    "BasisArbitrage": ("strategies.arbitrage.basis", "BasisArbitrage"),
    "OptionBoxArbitrage": ("strategies.arbitrage.option_box", "OptionBoxArbitrage"),
    "ConversionArbitrage": ("strategies.arbitrage.conversion", "ConversionArbitrage"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'strategies.arbitrage' has no attribute '{name}'")

    module_name, symbol_name = target
    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
