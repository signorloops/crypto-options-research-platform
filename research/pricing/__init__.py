"""
期权定价模型 (币本位 / Coin-margined)。

支持:
- 币本位期权定价 (Inverse Option Model)
- 币本位希腊字母计算
- Put-Call Parity (币本位版本)
"""
from research.pricing.inverse_options import InverseOptionPricer, inverse_option_parity
from research.pricing.rough_volatility import RoughVolConfig, RoughVolatilityPricer

__all__ = [
    "InverseOptionPricer",
    "inverse_option_parity",
    "RoughVolConfig",
    "RoughVolatilityPricer",
]
