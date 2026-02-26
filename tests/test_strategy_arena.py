"""Tests for StrategyArena reporting/comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import matplotlib
import pandas as pd
import pytest

from core.types import MarketState, OrderBook, OrderBookLevel, Position, QuoteAction
from research.backtest.arena import StrategyArena
from research.backtest.engine import BacktestResult
from strategies.base import MarketMakingStrategy

matplotlib.use("Agg")


def _make_backtest_result(name: str, pnl_values: list[float], sharpe: float) -> BacktestResult:
    index = pd.date_range("2026-01-01", periods=max(len(pnl_values), 1), freq="1D")
    pnl_series = (
        pd.Series(pnl_values, index=index[: len(pnl_values)])
        if pnl_values
        else pd.Series(dtype=float)
    )
    inventory_series = pd.Series([0.0] * len(pnl_series), index=pnl_series.index)
    crypto_balance_series = pd.Series([1.0] * len(pnl_series), index=pnl_series.index)
    total_pnl = float(pnl_values[-1]) if pnl_values else 0.0

    return BacktestResult(
        strategy_name=name,
        total_pnl_crypto=total_pnl / 50_000 if pnl_values else 0.0,
        total_pnl_usd=total_pnl,
        realized_pnl=total_pnl * 0.8,
        unrealized_pnl=total_pnl * 0.2,
        inventory_pnl=0.0,
        sharpe_ratio=sharpe,
        deflated_sharpe_ratio=0.0,
        max_drawdown=-0.1 if pnl_values else 0.0,
        volatility=0.2,
        sharpe_ci_95=(0.0, 1.0),
        drawdown_ci_95=(-0.2, 0.0),
        trade_count=max(0, len(pnl_values) - 1),
        buy_count=max(0, len(pnl_values) - 1) // 2,
        sell_count=max(0, len(pnl_values) - 1) // 2,
        avg_trade_size=1.0,
        avg_trade_pnl_crypto=0.001,
        total_spread_captured=10.0,
        avg_spread_captured_bps=1.5,
        inventory_cost=1.0,
        adverse_selection_cost=0.5,
        crypto_balance=1.0,
        crypto_balance_series=crypto_balance_series,
        pnl_series=pnl_series,
        inventory_series=inventory_series,
    )


@dataclass
class _DummyStrategy(MarketMakingStrategy):
    name: str
    reset_calls: int = 0

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        mid = state.order_book.mid_price or state.spot_price
        return QuoteAction(mid - 1.0, 1.0, mid + 1.0, 1.0)

    def get_internal_state(self) -> dict:
        return {"name": self.name}

    def reset(self) -> None:
        self.reset_calls += 1


def _market_data_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=20, freq="1D")
    return pd.DataFrame({"price": range(20)}, index=idx)


def test_scorecard_paths_and_report_generation():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)

    empty_result = _make_backtest_result("empty", [], sharpe=0.0)
    empty_score = arena._calculate_scorecard(empty_result)
    assert empty_score.total_pnl == 0
    assert empty_score.total_trades == 0

    rich_result = _make_backtest_result("rich", [0, 100, 80, 120, 150], sharpe=1.2)
    rich_score = arena._calculate_scorecard(rich_result)
    assert rich_score.total_pnl == 150
    assert "Strategy: rich" in rich_score.summary()

    arena.scorecards = {"empty": empty_score, "rich": rich_score}
    comparison = arena._create_comparison_df()
    assert set(["Strategy", "Sharpe", "Deflated Sharpe"]).issubset(comparison.columns)
    assert arena.get_winner("sharpe_ratio") == "rich"

    report = arena.generate_report()
    assert "STRATEGY ARENA - BACKTEST REPORT" in report
    assert "INDIVIDUAL STRATEGY RESULTS" in report


def test_statistical_comparison_and_plotting():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    r1 = _make_backtest_result("A", list(range(20)), sharpe=1.0)
    r2 = _make_backtest_result("B", [v * 0.8 for v in range(20)], sharpe=0.8)
    arena.scorecards = {"A": arena._calculate_scorecard(r1), "B": arena._calculate_scorecard(r2)}

    arena._apply_deflated_sharpe()
    assert arena.scorecards["A"].deflated_sharpe_ratio >= 0.0

    pvals = arena.statistical_comparison(correction="bonferroni")
    assert list(pvals.index) == ["A", "B"]
    assert pvals.shape == (2, 2)

    fig = arena.plot_comparison()
    assert len(fig.axes) == 9


def test_run_tournament_with_mocked_engine(monkeypatch):
    import research.backtest.arena as arena_module

    class _FakeEngine:
        def __init__(self, strategy, initial_crypto_balance, transaction_cost_bps):
            self.strategy = strategy

        def run(self, market_data):
            base = 100 if self.strategy.name == "S1" else 80
            return _make_backtest_result(self.strategy.name, [0, base, base + 10], sharpe=1.0)

    monkeypatch.setattr(arena_module, "BacktestEngine", _FakeEngine)

    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    s1 = _DummyStrategy(name="S1")
    s2 = _DummyStrategy(name="S2")

    comparison = arena.run_tournament([s1, s2], verbose=False)

    assert len(comparison) == 2
    assert set(comparison["Strategy"]) == {"S1", "S2"}
    assert s1.reset_calls == 1 and s2.reset_calls == 1


def test_rolling_sharpe_series_handles_short_and_long_inputs():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)

    short = pd.Series([100.0, 100.5], index=pd.date_range("2026-01-01", periods=2, freq="1D"))
    short_roll = arena._rolling_sharpe_series(short, window=5)
    assert short_roll.empty

    long = pd.Series(
        [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.8],
        index=pd.date_range("2026-01-01", periods=7, freq="1D"),
    )
    long_roll = arena._rolling_sharpe_series(long, window=3)
    assert len(long_roll) == len(long)
