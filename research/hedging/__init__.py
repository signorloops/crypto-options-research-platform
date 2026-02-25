"""
Hedging models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AdaptiveDeltaHedger",
    "AdaptiveHedgeConfig",
    "SimpleDeltaHedger",
    "DeepHedger",
    "DeepHedgingConfig",
    "DeepHedgingPolicy",
    "QuantoInverseHedger",
    "QuantoInverseHedgePlan",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AdaptiveDeltaHedger": ("research.hedging.adaptive_delta", "AdaptiveDeltaHedger"),
    "AdaptiveHedgeConfig": ("research.hedging.adaptive_delta", "AdaptiveHedgeConfig"),
    "SimpleDeltaHedger": ("research.hedging.adaptive_delta", "SimpleDeltaHedger"),
    "DeepHedger": ("research.hedging.deep_hedging", "DeepHedger"),
    "DeepHedgingConfig": ("research.hedging.deep_hedging", "DeepHedgingConfig"),
    "DeepHedgingPolicy": ("research.hedging.deep_hedging", "DeepHedgingPolicy"),
    "QuantoInverseHedger": ("research.hedging.quanto_inverse", "QuantoInverseHedger"),
    "QuantoInverseHedgePlan": ("research.hedging.quanto_inverse", "QuantoInverseHedgePlan"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'research.hedging' has no attribute '{name}'")

    module_name, symbol_name = target
    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
