"""
套利策略模块。

包含:
- 跨交易所套利 (Cross-exchange arbitrage)
- 期现套利 (Basis trading / Cash-and-carry)
- 期权盒式套利 (Option box spread)
- 转换套利 (Conversion/Reversal arbitrage)
"""
from strategies.arbitrage.basis import BasisArbitrage
from strategies.arbitrage.conversion import ConversionArbitrage
from strategies.arbitrage.cross_exchange import CrossExchangeArbitrage
from strategies.arbitrage.option_box import OptionBoxArbitrage

__all__ = [
    "CrossExchangeArbitrage",
    "BasisArbitrage",
    "OptionBoxArbitrage",
    "ConversionArbitrage",
]
