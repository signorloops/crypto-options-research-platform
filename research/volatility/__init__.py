"""
波动率分析模块。

包含历史波动率、隐含波动率和波动率预测模型。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Historical volatility
    "realized_variance",
    "realized_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    # Models
    "ewma_volatility",
    "garch_volatility",
    "har_volatility",
    "bipower_variation",
    "medrv_volatility",
    "two_scale_realized_volatility",
    "realized_kernel_volatility",
    "egarch_volatility",
    "gjr_garch_volatility",
    "hamilton_filter_regime_switching",
    # Implied volatility
    "implied_volatility",
    "implied_volatility_bisection",
    "implied_volatility_newton",
    "implied_volatility_jaeckel",
    "implied_volatility_lbr",
    "black_scholes_price",
    "VolatilitySurface",
    "SVIParams",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "realized_variance": ("research.volatility.historical", "realized_variance"),
    "realized_volatility": ("research.volatility.historical", "realized_volatility"),
    "parkinson_volatility": ("research.volatility.historical", "parkinson_volatility"),
    "garman_klass_volatility": ("research.volatility.historical", "garman_klass_volatility"),
    "rogers_satchell_volatility": ("research.volatility.historical", "rogers_satchell_volatility"),
    "yang_zhang_volatility": ("research.volatility.historical", "yang_zhang_volatility"),
    "ewma_volatility": ("research.volatility.models", "ewma_volatility"),
    "garch_volatility": ("research.volatility.models", "garch_volatility"),
    "har_volatility": ("research.volatility.models", "har_volatility"),
    "bipower_variation": ("research.volatility.models", "bipower_variation"),
    "medrv_volatility": ("research.volatility.models", "medrv_volatility"),
    "two_scale_realized_volatility": (
        "research.volatility.models",
        "two_scale_realized_volatility",
    ),
    "realized_kernel_volatility": ("research.volatility.models", "realized_kernel_volatility"),
    "egarch_volatility": ("research.volatility.models", "egarch_volatility"),
    "gjr_garch_volatility": ("research.volatility.models", "gjr_garch_volatility"),
    "hamilton_filter_regime_switching": (
        "research.volatility.models",
        "hamilton_filter_regime_switching",
    ),
    "implied_volatility": ("research.volatility.implied", "implied_volatility"),
    "implied_volatility_bisection": ("research.volatility.implied", "implied_volatility_bisection"),
    "implied_volatility_newton": ("research.volatility.implied", "implied_volatility_newton"),
    "implied_volatility_jaeckel": ("research.volatility.implied", "implied_volatility_jaeckel"),
    "implied_volatility_lbr": ("research.volatility.implied", "implied_volatility_lbr"),
    "black_scholes_price": ("research.volatility.implied", "black_scholes_price"),
    "VolatilitySurface": ("research.volatility.implied", "VolatilitySurface"),
    "SVIParams": ("research.volatility.implied", "SVIParams"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'research.volatility' has no attribute '{name}'")

    module_name, symbol_name = target
    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
