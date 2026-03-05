"""Strategy arena for fair backtest comparison across multiple strategies."""

from dataclasses import dataclass, field
from datetime import datetime
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from research.backtest.engine import BacktestEngine, BacktestResult
from strategies.base import MarketMakingStrategy
from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


def _pairwise_p_value(returns1: pd.Series, returns2: pd.Series) -> float:
    """Robust pairwise p-value with deterministic handling for degenerate samples."""
    from scipy import stats
    values1 = returns1.to_numpy(dtype=float)
    values2 = returns2.to_numpy(dtype=float)
    values1 = values1[np.isfinite(values1)]
    values2 = values2[np.isfinite(values2)]
    if len(values1) < 2 or len(values2) < 2:
        return 1.0
    if np.allclose(values1, values2, rtol=1e-10, atol=1e-12):
        return 1.0
    std1 = float(np.std(values1, ddof=1))
    std2 = float(np.std(values2, ddof=1))
    mean_diff = abs(float(np.mean(values1) - np.mean(values2)))
    mean_scale = max(abs(float(np.mean(values1))), abs(float(np.mean(values2))), 1.0)
    tol = 1e-12 * mean_scale
    # Degenerate distributions with effectively zero variance.
    if std1 <= 1e-12 and std2 <= 1e-12:
        return 1.0 if mean_diff <= tol else 0.0
    if std1 <= 1e-12 or std2 <= 1e-12:
        if mean_diff <= tol:
            return 1.0
        _, p_value = stats.mannwhitneyu(values1, values2, alternative="two-sided")
        return float(p_value)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            _, p_value = stats.ttest_ind(values1, values2, equal_var=False)
            return float(p_value)
        except RuntimeWarning:
            # Fall back to rank-based test when moment-based t-test is unstable.
            _, p_value = stats.mannwhitneyu(values1, values2, alternative="two-sided")
            return float(p_value)


@dataclass
class StrategyScorecard:
    """Comprehensive performance metrics for a strategy."""

    strategy_name: str

    # Return metrics
    total_pnl: float
    total_return_pct: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade metrics
    total_trades: int
    win_rate: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Market making specific
    spread_capture: float
    adverse_selection_cost: float
    inventory_cost: float
    fill_rate: float

    # Consistency
    daily_pnl_std: float
    worst_day: float
    best_day: float
    deflated_sharpe_ratio: float = 0.0

    # Time series for plotting
    pnl_series: pd.Series = field(default_factory=lambda: pd.Series())
    drawdown_series: pd.Series = field(default_factory=lambda: pd.Series())
    inventory_series: pd.Series = field(default_factory=lambda: pd.Series())

    def summary(self) -> str:
        """Generate formatted summary."""
        return f"""
{'='*60}
Strategy: {self.strategy_name}
{'='*60}
Returns:
  Total PnL:          ${self.total_pnl:,.2f} ({self.total_return_pct:.2%})
  Annualized Return:  {self.annualized_return:.2%}
  Sharpe Ratio:       {self.sharpe_ratio:.2f}
  Deflated Sharpe:    {self.deflated_sharpe_ratio:.2f}
  Sortino Ratio:      {self.sortino_ratio:.2f}

Risk:
  Max Drawdown:       {self.max_drawdown:.2%}
  Calmar Ratio:       {self.calmar_ratio:.2f}
  Daily PnL Std:      ${self.daily_pnl_std:,.2f}

Trading:
  Total Trades:       {self.total_trades}
  Win Rate:           {self.win_rate:.1%}
  Avg Trade PnL:      ${self.avg_trade_pnl:.2f}
  Profit Factor:      {self.profit_factor:.2f}

Market Making:
  Spread Capture:     ${self.spread_capture:,.2f}
  Adverse Select:     ${self.adverse_selection_cost:,.2f}
  Inventory Cost:     ${self.inventory_cost:,.2f}
  Fill Rate:          {self.fill_rate:.1%}
{'='*60}
"""


def _build_empty_scorecard(result: BacktestResult) -> StrategyScorecard:
    return StrategyScorecard(
        strategy_name=result.strategy_name,
        total_pnl=0,
        total_return_pct=0,
        annualized_return=0,
        sharpe_ratio=0,
        deflated_sharpe_ratio=0,
        sortino_ratio=0,
        max_drawdown=0,
        calmar_ratio=0,
        total_trades=0,
        win_rate=0,
        avg_trade_pnl=0,
        avg_win=0,
        avg_loss=0,
        profit_factor=0,
        spread_capture=0,
        adverse_selection_cost=0,
        inventory_cost=0,
        fill_rate=0,
        daily_pnl_std=0,
        worst_day=0,
        best_day=0,
    )


def _compute_sortino_ratio(daily_returns: pd.Series, periods_per_year: float) -> float:
    target_return = 0.0
    downside_returns = daily_returns[daily_returns < target_return]
    if len(downside_returns) == 0:
        return 0.0

    downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    if downside_deviation <= 0:
        return 0.0

    excess_return = daily_returns.mean() - target_return
    return float((excess_return / downside_deviation) * np.sqrt(periods_per_year))


def _compute_trade_stats(
    trade_count: int, daily_returns: pd.Series
) -> Tuple[float, float, float, float]:
    if trade_count <= 0 or len(daily_returns) == 0:
        return 0.0, 0.0, 0.0, 0.0

    wins_pnl = daily_returns[daily_returns > 0]
    losses_pnl = daily_returns[daily_returns < 0]
    win_rate = float((daily_returns > 0).sum() / len(daily_returns))

    avg_win = float(wins_pnl.mean()) if len(wins_pnl) > 0 else 0.0
    avg_loss = float(abs(losses_pnl.mean())) if len(losses_pnl) > 0 else 0.0
    total_gains = float(wins_pnl.sum()) if len(wins_pnl) > 0 else 0.0
    total_losses = float(abs(losses_pnl.sum())) if len(losses_pnl) > 0 else 0.0
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")
    return win_rate, avg_win, avg_loss, profit_factor


def _compute_drawdown_series(pnl_series: pd.Series, initial_capital: float) -> pd.Series:
    running_max = pnl_series.expanding().max()
    return (pnl_series - running_max) / (running_max + initial_capital)


def _annualized_return_from_total(
    *, total_return_pct: float, periods_per_year: float, periods_observed: int
) -> float:
    if total_return_pct <= -1:
        return -1.0
    return float((1 + total_return_pct) ** (periods_per_year / periods_observed) - 1)


def _daily_pnl_stats(daily_returns: pd.Series) -> tuple[float, float, float]:
    if len(daily_returns) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(daily_returns.std()),
        float(daily_returns.min()),
        float(daily_returns.max()),
    )


def _calmar_ratio(
    *, annualized_return: float, max_drawdown: float, min_drawdown_threshold: float = 0.001
) -> float:
    if abs(max_drawdown) <= min_drawdown_threshold:
        return 0.0
    return float(annualized_return / abs(max_drawdown))


def _build_scorecard_from_result(
    *,
    result: BacktestResult,
    pnl_series: pd.Series,
    drawdown_series: pd.Series,
    metrics: dict[str, float],
) -> StrategyScorecard:
    return StrategyScorecard(
        strategy_name=result.strategy_name,
        total_pnl=metrics["total_pnl"],
        total_return_pct=metrics["total_return_pct"],
        annualized_return=metrics["annualized_return"],
        sharpe_ratio=metrics["sharpe"],
        deflated_sharpe_ratio=metrics["deflated_sharpe"],
        sortino_ratio=metrics["sortino"],
        max_drawdown=result.max_drawdown,
        calmar_ratio=metrics["calmar"],
        total_trades=result.trade_count,
        win_rate=metrics["win_rate"],
        avg_trade_pnl=result.avg_trade_pnl_crypto,
        avg_win=metrics["avg_win"],
        avg_loss=metrics["avg_loss"],
        profit_factor=metrics["profit_factor"],
        spread_capture=result.total_spread_captured or 0,
        adverse_selection_cost=result.adverse_selection_cost or 0,
        inventory_cost=result.inventory_cost or 0,
        fill_rate=0.3,
        daily_pnl_std=metrics["daily_pnl_std"],
        worst_day=metrics["worst_day"],
        best_day=metrics["best_day"],
        pnl_series=pnl_series,
        drawdown_series=drawdown_series,
        inventory_series=result.inventory_series,
    )


class StrategyArena:
    """Fair comparison framework for market making strategies."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 2.0,
        annualization_periods: Optional[float] = None,
    ):
        self.market_data = market_data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.annualization_periods = annualization_periods

        self.results: Dict[str, BacktestResult] = {}
        self.scorecards: Dict[str, StrategyScorecard] = {}

    def _periods_per_year(self, series: pd.Series) -> float:
        """Infer annualization periods from series frequency, with optional override."""
        if self.annualization_periods is not None and self.annualization_periods > 0:
            return float(self.annualization_periods)
        if len(series) <= 1:
            return 365.25
        if isinstance(series.index, pd.DatetimeIndex):
            deltas = series.index.to_series().diff().dropna().dt.total_seconds()
            if len(deltas) > 0:
                step_seconds = float(deltas.median())
                if step_seconds > 0:
                    return (365.25 * 24.0 * 3600.0) / step_seconds
        return 365.25

    def run_tournament(
        self, strategies: List[MarketMakingStrategy], verbose: bool = True
    ) -> pd.DataFrame:
        """Run all strategies and return a comparison DataFrame."""
        self.results = {}
        self.scorecards = {}

        for strategy in strategies:
            if verbose:
                logger.info("Running strategy", extra=log_extra(strategy=strategy.name))

            strategy.reset()
            engine = BacktestEngine(
                strategy=strategy,
                initial_crypto_balance=self.initial_capital,
                transaction_cost_bps=self.transaction_cost_bps,
            )
            result = engine.run(self.market_data)

            self.results[strategy.name] = result

            scorecard = self._calculate_scorecard(result)
            self.scorecards[strategy.name] = scorecard

            if verbose:
                logger.info(
                    "Strategy results",
                    extra=log_extra(
                        strategy=strategy.name,
                        pnl=result.total_pnl_usd,
                        trades=result.trade_count,
                        sharpe=result.sharpe_ratio,
                    ),
                )

        self._apply_deflated_sharpe()
        return self._create_comparison_df()

    def _apply_deflated_sharpe(self) -> None:
        """Apply multiple-testing correction to Sharpe across all compared strategies."""
        n_trials = max(1, len(self.scorecards))
        for sc in self.scorecards.values():
            n_obs = max(len(sc.pnl_series.diff().dropna()), 1)
            if n_obs < 5:
                sc.deflated_sharpe_ratio = 0.0
                continue
            expected_max_sr = norm.ppf(1.0 - 1.0 / max(n_trials, 2)) / np.sqrt(max(n_obs - 1, 1))
            denom = max(np.sqrt(1.0 / max(n_obs - 1, 1)), 1e-12)
            z = (sc.sharpe_ratio - expected_max_sr) / denom
            sc.deflated_sharpe_ratio = float(norm.cdf(z))

    def _calculate_scorecard(self, result: BacktestResult) -> StrategyScorecard:
        """Calculate comprehensive metrics from backtest result."""
        pnl_series = result.pnl_series
        if len(pnl_series) == 0:
            return _build_empty_scorecard(result)
        total_pnl = result.total_pnl_usd
        total_return_pct = total_pnl / self.initial_capital
        periods_per_year = self._periods_per_year(pnl_series)
        periods_observed = max(len(pnl_series) - 1, 1)
        annualized_return = _annualized_return_from_total(total_return_pct=total_return_pct, periods_per_year=periods_per_year, periods_observed=periods_observed)
        daily_returns = pnl_series.diff().dropna()
        sharpe = result.sharpe_ratio
        deflated_sharpe = getattr(result, "deflated_sharpe_ratio", 0.0)
        sortino = _compute_sortino_ratio(daily_returns, self._periods_per_year(daily_returns))
        calmar = _calmar_ratio(annualized_return=annualized_return, max_drawdown=result.max_drawdown)
        win_rate, avg_win, avg_loss, profit_factor = _compute_trade_stats(result.trade_count, daily_returns)
        daily_pnl_std, worst_day, best_day = _daily_pnl_stats(daily_returns)
        drawdown_series = _compute_drawdown_series(pnl_series, self.initial_capital)
        return _build_scorecard_from_result(
            result=result,
            pnl_series=pnl_series,
            drawdown_series=drawdown_series,
            metrics={
                "total_pnl": total_pnl,
                "total_return_pct": total_return_pct,
                "annualized_return": annualized_return,
                "sharpe": sharpe,
                "deflated_sharpe": deflated_sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "daily_pnl_std": daily_pnl_std,
                "worst_day": worst_day,
                "best_day": best_day,
            },
        )

    def _create_comparison_df(self) -> pd.DataFrame:
        """Create comparison DataFrame from scorecards."""
        rows = []
        for name, sc in self.scorecards.items():
            rows.append(
                {
                    "Strategy": name,
                    "Total PnL ($)": sc.total_pnl,
                    "Return (%)": sc.total_return_pct * 100,
                    "Annual Return (%)": sc.annualized_return * 100,
                    "Sharpe": sc.sharpe_ratio,
                    "Deflated Sharpe": sc.deflated_sharpe_ratio,
                    "Sortino": sc.sortino_ratio,
                    "Max DD (%)": sc.max_drawdown * 100,
                    "Calmar": sc.calmar_ratio,
                    "Trades": sc.total_trades,
                    "Avg Trade ($)": sc.avg_trade_pnl,
                    "Daily Std ($)": sc.daily_pnl_std,
                }
            )

        return pd.DataFrame(rows)

    def _plot_cumulative_pnl(self, ax: plt.Axes) -> None:
        for name, sc in self.scorecards.items():
            if len(sc.pnl_series) > 0:
                ax.plot(sc.pnl_series.index, sc.pnl_series.values, label=name)
        ax.set_title("Cumulative PnL")
        ax.set_ylabel("PnL ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, ax: plt.Axes) -> None:
        for name, sc in self.scorecards.items():
            if len(sc.drawdown_series) > 0:
                ax.fill_between(
                    sc.drawdown_series.index, sc.drawdown_series.values, 0, alpha=0.3, label=name
                )
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_inventory(self, ax: plt.Axes) -> None:
        for name, sc in self.scorecards.items():
            if len(sc.inventory_series) > 0:
                ax.plot(
                    sc.inventory_series.index, sc.inventory_series.values, label=name, alpha=0.7
                )
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_title("Inventory Position")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_total_return_bars(self, ax: plt.Axes) -> None:
        strategies = list(self.scorecards.keys())
        returns = [self.scorecards[s].total_return_pct * 100 for s in strategies]
        colors = ["green" if r > 0 else "red" for r in returns]
        ax.bar(strategies, returns, color=colors, alpha=0.7)
        ax.set_title("Total Return (%)")
        ax.set_ylabel("Return (%)")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_risk_return_scatter(self, ax: plt.Axes) -> None:
        for name, sc in self.scorecards.items():
            ax.scatter(
                sc.max_drawdown * 100, sc.annualized_return * 100, s=200, alpha=0.6, label=name
            )
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Risk-Return Profile")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_sharpe_bars(self, ax: plt.Axes) -> None:
        strategies = list(self.scorecards.keys())
        sharpes = [self.scorecards[s].sharpe_ratio for s in strategies]
        colors = ["green" if s > 1 else "orange" if s > 0 else "red" for s in sharpes]
        ax.bar(strategies, sharpes, color=colors, alpha=0.7)
        ax.axhline(y=1, color="g", linestyle="--", alpha=0.5, label="Good (>1)")
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_title("Sharpe Ratio")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_pnl_distribution(self, ax: plt.Axes) -> None:
        for name, sc in self.scorecards.items():
            if len(sc.pnl_series) <= 1:
                continue
            daily_pnls = sc.pnl_series.diff().dropna()
            values = daily_pnls.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            if np.ptp(values) <= 1e-12:
                center = float(values[0])
                span = max(1e-6, abs(center) * 0.01)
                bins = [center - span, center + span]
            else:
                bins = 30
            ax.hist(values, bins=bins, alpha=0.5, label=name, density=True)
        ax.set_title("Daily PnL Distribution")
        ax.set_xlabel("Daily PnL ($)")
        ax.legend()

    def _rolling_sharpe_series(self, pnl_series: pd.Series, window: int) -> pd.Series:
        if len(pnl_series) <= window:
            return pd.Series(dtype=float)
        daily_returns = pnl_series.diff()
        annualization = np.sqrt(self._periods_per_year(daily_returns.dropna()))
        return (
            daily_returns.rolling(window).mean()
            / daily_returns.rolling(window).std()
            * annualization
        )

    def _plot_rolling_sharpe(self, ax: plt.Axes, window: int) -> None:
        for name, sc in self.scorecards.items():
            rolling_sharpe = self._rolling_sharpe_series(sc.pnl_series, window=window)
            if rolling_sharpe.empty:
                continue
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, alpha=0.7)
        ax.set_title(f"Rolling Sharpe ({window} periods)")
        ax.axhline(y=1, color="g", linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metrics_table(self, ax: plt.Axes) -> None:
        ax.axis("off")

        comparison = self._create_comparison_df()
        table_data = []
        for _, row in comparison.iterrows():
            table_data.append(
                [
                    row["Strategy"][:15],
                    f"${row['Total PnL ($)']:,.0f}",
                    f"{row['Return (%)']:.1f}%",
                    f"{row['Sharpe']:.2f}",
                    f"{row['Max DD (%)']:.1f}%",
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=["Strategy", "PnL", "Return", "Sharpe", "Max DD"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

    def plot_comparison(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Generate comparison plots."""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        window = min(50, len(self.market_data) // 10)
        self._plot_cumulative_pnl(axes[0, 0])
        self._plot_drawdown(axes[0, 1])
        self._plot_inventory(axes[0, 2])
        self._plot_total_return_bars(axes[1, 0])
        self._plot_risk_return_scatter(axes[1, 1])
        self._plot_sharpe_bars(axes[1, 2])
        self._plot_pnl_distribution(axes[2, 0])
        self._plot_rolling_sharpe(axes[2, 1], window=window)
        self._plot_metrics_table(axes[2, 2])

        plt.tight_layout()
        return fig

    def get_winner(self, metric: str = "sharpe_ratio") -> str:
        """Get the best-performing strategy by a given metric."""
        if not self.scorecards:
            raise ValueError("No results available. Run tournament first.")

        rankings = sorted(
            self.scorecards.items(),
            key=lambda x: getattr(x[1], metric, float("-inf")),
            reverse=True,
        )

        return rankings[0][0]

    def statistical_comparison(self, correction: str = "bonferroni") -> pd.DataFrame:
        """Perform pairwise statistical tests between strategies."""

        names = list(self.scorecards.keys())
        n = len(names)

        if n < 2:
            return pd.DataFrame()

        p_values = np.ones((n, n))
        raw_p_values = []
        indices = []

        for i in range(n):
            for j in range(i + 1, n):
                sc1 = self.scorecards[names[i]]
                sc2 = self.scorecards[names[j]]

                if len(sc1.pnl_series) > 10 and len(sc2.pnl_series) > 10:
                    returns1 = sc1.pnl_series.diff().dropna()
                    returns2 = sc2.pnl_series.diff().dropna()

                    p_value = _pairwise_p_value(returns1, returns2)
                    p_values[i, j] = p_value
                    p_values[j, i] = p_value
                    raw_p_values.append(p_value)
                    indices.append((i, j))

        n_tests = len(raw_p_values)
        if n_tests > 1 and correction == "bonferroni":
            for (i, j), p in zip(indices, raw_p_values):
                corrected_p = min(p * n_tests, 1.0)
                p_values[i, j] = corrected_p
                p_values[j, i] = corrected_p

        return pd.DataFrame(p_values, index=names, columns=names)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a text report and optionally persist it to disk."""
        lines = []
        lines.append("=" * 70)
        lines.append("STRATEGY ARENA - BACKTEST REPORT")
        lines.append("=" * 70)
        lines.append(f"\nBacktest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        lines.append(f"Transaction Cost: {self.transaction_cost_bps} bps")
        if len(self.market_data) > 0:
            start_date = self.market_data.index[0]
            end_date = self.market_data.index[-1]
            lines.append(f"Data Period: {start_date} to {end_date}")
        else:
            lines.append("Data Period: No data")
        lines.append(f"Number of Strategies: {len(self.scorecards)}")
        lines.append("\n" + "=" * 70)
        lines.append("INDIVIDUAL STRATEGY RESULTS")
        lines.append("=" * 70)
        for name, sc in self.scorecards.items():
            lines.append(sc.summary())
        lines.append("\n" + "=" * 70)
        lines.append("STRATEGY RANKINGS")
        lines.append("=" * 70)
        metrics = ["total_pnl", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]
        for metric in metrics:
            winner = self.get_winner(metric)
            value = getattr(self.scorecards[winner], metric)
            lines.append(f"\nBest by {metric}: {winner} ({value:.4f})")
        report = "\n".join(lines)
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
        return report
