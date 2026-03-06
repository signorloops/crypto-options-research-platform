"""Tests for StrategyArena reporting/comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import warnings

import matplotlib
import pandas as pd
import pytest

from core.types import MarketState, OrderBook, OrderBookLevel, Position, QuoteAction
import research.backtest.arena as arena_module
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


def test_arena_report_body_lines_include_results_and_rankings():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    rich_result = _make_backtest_result("rich", [0, 100, 80, 120, 150], sharpe=1.2)
    arena.scorecards = {"rich": arena._calculate_scorecard(rich_result)}

    lines = arena_module._arena_report_body_lines(arena)

    assert lines[1] == "INDIVIDUAL STRATEGY RESULTS"
    assert any("Strategy: rich" in line for line in lines)
    assert "STRATEGY RANKINGS" in lines


def test_scorecard_summary_sections_include_named_blocks():
    scorecard = arena_module.StrategyScorecard(
        strategy_name="rich",
        total_pnl=150.0,
        total_return_pct=0.0015,
        annualized_return=0.12,
        sharpe_ratio=1.2,
        sortino_ratio=1.3,
        max_drawdown=-0.1,
        calmar_ratio=1.2,
        total_trades=4,
        win_rate=0.75,
        avg_trade_pnl=0.001,
        avg_win=60.0,
        avg_loss=20.0,
        profit_factor=3.0,
        spread_capture=10.0,
        adverse_selection_cost=0.5,
        inventory_cost=1.0,
        fill_rate=0.3,
        daily_pnl_std=45.0,
        worst_day=-20.0,
        best_day=100.0,
    )

    lines = arena_module._scorecard_summary_sections(scorecard)

    assert "Strategy: rich" in lines[1]
    assert "Returns:" in lines
    assert "Risk:" in lines
    assert "Trading:" in lines
    assert "Market Making:" in lines


def test_scorecard_section_helpers_render_titles_and_lines():
    scorecard = arena_module.StrategyScorecard(
        strategy_name="rich",
        total_pnl=150.0,
        total_return_pct=0.0015,
        annualized_return=0.12,
        sharpe_ratio=1.2,
        sortino_ratio=1.3,
        max_drawdown=-0.1,
        calmar_ratio=1.2,
        total_trades=4,
        win_rate=0.75,
        avg_trade_pnl=0.001,
        avg_win=60.0,
        avg_loss=20.0,
        profit_factor=3.0,
        spread_capture=10.0,
        adverse_selection_cost=0.5,
        inventory_cost=1.0,
        fill_rate=0.3,
        daily_pnl_std=45.0,
        worst_day=-20.0,
        best_day=100.0,
    )

    sections = arena_module._scorecard_section_specs(scorecard)
    rendered = arena_module._render_scorecard_section(*sections[0])

    assert sections[0][0] == "Returns:"
    assert any("Total PnL" in line for line in sections[0][1])
    assert rendered[0] == "Returns:"
    assert any("Sharpe Ratio" in line for line in rendered)


def test_scorecard_payload_helpers_return_expected_defaults_and_fields():
    result = _make_backtest_result("rich", [0, 100, 80, 120, 150], sharpe=1.2)
    metrics = {
        "total_pnl": 150.0,
        "total_return_pct": 0.0015,
        "annualized_return": 0.12,
        "sharpe": 1.2,
        "deflated_sharpe": 0.5,
        "sortino": 1.3,
        "calmar": 1.2,
        "win_rate": 0.75,
        "avg_win": 60.0,
        "avg_loss": 20.0,
        "profit_factor": 3.0,
        "daily_pnl_std": 45.0,
        "worst_day": -20.0,
        "best_day": 100.0,
    }

    empty = arena_module._empty_scorecard_fields("empty")
    payload = arena_module._scorecard_result_fields(result, metrics)

    assert empty["strategy_name"] == "empty"
    assert empty["total_pnl"] == 0
    assert empty["profit_factor"] == 0
    assert payload["strategy_name"] == "rich"
    assert payload["total_pnl"] == 150.0
    assert payload["avg_trade_pnl"] == result.avg_trade_pnl_crypto
    assert payload["spread_capture"] == 10.0
    assert payload["fill_rate"] == 0.3


def test_scorecard_summary_metrics_helper_matches_expected_values():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    result = _make_backtest_result("rich", [0, 100, 80, 120, 150], sharpe=1.2)
    pnl_series = result.pnl_series
    daily_returns = pnl_series.diff().dropna()

    metrics = arena_module._scorecard_summary_metrics(
        result=result,
        initial_capital=arena.initial_capital,
        periods_per_year=arena._periods_per_year(pnl_series),
        periods_observed=max(len(pnl_series) - 1, 1),
        daily_returns=daily_returns,
    )

    assert metrics["total_pnl"] == 150.0
    assert metrics["total_return_pct"] == pytest.approx(0.0015)
    assert metrics["sharpe"] == 1.2
    assert metrics["win_rate"] == pytest.approx(0.75)
    assert metrics["profit_factor"] > 1.0
    assert metrics["best_day"] == 100.0


def test_scorecard_metric_helpers_split_return_and_trade_components():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    result = _make_backtest_result("rich", [0, 100, 80, 120, 150], sharpe=1.2)
    pnl_series = result.pnl_series
    daily_returns = pnl_series.diff().dropna()

    return_metrics = arena_module._scorecard_return_metrics(
        result=result,
        initial_capital=arena.initial_capital,
        periods_per_year=arena._periods_per_year(pnl_series),
        periods_observed=max(len(pnl_series) - 1, 1),
    )
    trade_metrics = arena_module._scorecard_trade_metrics(
        result=result,
        daily_returns=daily_returns,
        periods_per_year=arena._periods_per_year(pnl_series),
    )

    assert return_metrics["total_pnl"] == 150.0
    assert return_metrics["annualized_return"] > 0.0
    assert trade_metrics["win_rate"] == pytest.approx(0.75)
    assert trade_metrics["profit_factor"] > 1.0
    assert trade_metrics["best_day"] == 100.0


def test_comparison_helpers_format_rows_and_metric_values():
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    rich_result = _make_backtest_result("rich", [0, 100, 80, 120, 150], sharpe=1.2)
    rich_score = arena._calculate_scorecard(rich_result)

    row = arena_module._comparison_row("rich", rich_score)
    strategies, sharpes = arena_module._scorecard_metric_values(
        {"rich": rich_score},
        "sharpe_ratio",
    )
    table_rows = arena_module._comparison_table_rows(pd.DataFrame([row]))

    assert row["Strategy"] == "rich"
    assert row["Total PnL ($)"] == 150.0
    assert row["Return (%)"] == pytest.approx(0.15)
    assert strategies == ["rich"]
    assert sharpes == [1.2]
    assert table_rows == [["rich", "$150", "0.1%", "1.20", "-10.0%"]]


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


def test_run_single_strategy_returns_result_and_scorecard(monkeypatch):
    class _FakeEngine:
        def __init__(self, strategy, initial_crypto_balance, transaction_cost_bps):
            self.strategy = strategy

        def run(self, market_data):
            return _make_backtest_result(self.strategy.name, [0, 90, 110], sharpe=0.9)

    monkeypatch.setattr(arena_module, "BacktestEngine", _FakeEngine)

    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    strategy = _DummyStrategy(name="Solo")

    result, scorecard = arena._run_single_strategy(strategy, verbose=False)

    assert strategy.reset_calls == 1
    assert result.strategy_name == "Solo"
    assert scorecard.strategy_name == "Solo"
    assert scorecard.total_pnl == 110.0


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


def test_annualization_periods_infers_from_series_frequency():
    """Arena annualization should adapt to intraday vs daily timestamps."""
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)

    daily = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2026-01-01", periods=3, freq="1D"))
    hourly = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2026-01-01", periods=3, freq="1h"))

    daily_periods = arena._periods_per_year(daily)
    hourly_periods = arena._periods_per_year(hourly)

    assert daily_periods == pytest.approx(365.25, rel=0.05)
    assert hourly_periods == pytest.approx(365.25 * 24.0, rel=0.05)
    assert hourly_periods > daily_periods


def test_statistical_comparison_avoids_runtime_warning_on_near_identical_series():
    """Near-identical return samples should not emit scipy precision RuntimeWarning."""
    arena = StrategyArena(_market_data_frame(), initial_capital=100000.0)
    # Nearly identical daily PnL paths trigger scipy catastrophic-cancellation warnings
    # unless handled explicitly.
    pnl_a = [float(v) for v in range(0, 25)]
    pnl_b = [float(v) + 1e-10 for v in range(0, 25)]
    r1 = _make_backtest_result("A", pnl_a, sharpe=1.0)
    r2 = _make_backtest_result("B", pnl_b, sharpe=1.0)
    arena.scorecards = {"A": arena._calculate_scorecard(r1), "B": arena._calculate_scorecard(r2)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pvals = arena.statistical_comparison(correction="bonferroni")

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert pvals.shape == (2, 2)
