"""
Signal and regime detection models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BOCDConfig",
    "OnlineBayesianChangepointDetector",
    "JumpRiskPremiaEstimator",
    "JumpRiskPremiaSignal",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BOCDConfig": ("research.signals.bayesian_changepoint", "BOCDConfig"),
    "OnlineBayesianChangepointDetector": (
        "research.signals.bayesian_changepoint",
        "OnlineBayesianChangepointDetector",
    ),
    "JumpRiskPremiaEstimator": ("research.signals.jump_risk_premia", "JumpRiskPremiaEstimator"),
    "JumpRiskPremiaSignal": ("research.signals.jump_risk_premia", "JumpRiskPremiaSignal"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'research.signals' has no attribute '{name}'")

    module_name, symbol_name = target
    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
