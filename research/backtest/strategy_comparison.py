"""Utility helpers for comparing multiple strategies on the same dataset."""

from typing import Dict, List

import pandas as pd

from research.backtest.engine import BacktestEngine
from strategies.base import MarketMakingStrategy


class StrategyComparison:
    """Run multiple strategies on shared market data and collect comparable metrics."""

    def __init__(self, strategies: List[MarketMakingStrategy]):
        self.strategies = strategies
        self.results: Dict[str, Dict] = {}

    def run_comparison(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Run each strategy in the same backtest environment and aggregate results."""
        results = []
        for strategy in self.strategies:
            strategy.reset()
            engine = BacktestEngine(strategy)
            result = engine.run(market_data)

            results.append(
                {
                    "strategy": strategy.name,
                    "total_pnl": result.total_pnl,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "trade_count": result.trade_count,
                    "avg_trade_pnl": result.avg_trade_pnl,
                    "inventory_cost": result.inventory_cost,
                    "adverse_selection_cost": result.adverse_selection_cost,
                }
            )

        return pd.DataFrame(results)
