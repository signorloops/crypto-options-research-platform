"""
波动率分析模块。

包含历史波动率、隐含波动率和波动率预测模型。
"""
from research.volatility.historical import (
    garman_klass_volatility,
    parkinson_volatility,
    realized_variance,
    realized_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
)
from research.volatility.implied import (
    SVIParams,
    VolatilitySurface,
    black_scholes_price,
    implied_volatility,
    implied_volatility_bisection,
    implied_volatility_jaeckel,
    implied_volatility_lbr,
    implied_volatility_newton,
)
from research.volatility.models import (
    bipower_variation,
    ewma_volatility,
    egarch_volatility,
    garch_volatility,
    gjr_garch_volatility,
    har_volatility,
    hamilton_filter_regime_switching,
    medrv_volatility,
    realized_kernel_volatility,
    two_scale_realized_volatility,
)

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
