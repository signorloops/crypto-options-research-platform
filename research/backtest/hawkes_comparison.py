"""
Hawkes Market Making Strategy - Comprehensive Backtest Comparison Framework.

This module provides specialized backtesting and comparison capabilities for
Hawkes-based market making strategies against various benchmarks.

实验设计包含三个场景:
- 场景 A: 合成 Hawkes 数据（验证理论优势）
- 场景 B: 真实历史数据
- 场景 C: 压力测试
"""
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from data.generators.hawkes import HawkesProcess, HawkesParameters
from research.backtest.engine import BacktestEngine, BacktestResult
from research.backtest.arena import StrategyArena, StrategyScorecard
from strategies.base import MarketMakingStrategy
from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


class ScenarioType(Enum):
    """Types of test scenarios."""
    SYNTHETIC_HAWKES = "synthetic_hawkes"
    REAL_HISTORICAL = "real_historical"
    STRESS_TEST = "stress_test"


@dataclass
class HawkesScenarioConfig:
    """Configuration for Hawkes process scenarios."""
    name: str
    mu: float = 0.1
    alpha: float = 0.4
    beta: float = 0.8
    T: float = 86400.0  # 1 day in seconds
    seed: int = 42

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def clustering_level(self) -> str:
        """Categorize clustering level based on branching ratio."""
        br = self.branching_ratio
        if br < 0.2:
            return "low"
        elif br < 0.6:
            return "medium"
        elif br < 0.9:
            return "high"
        else:
            return "critical"


class ScenarioGenerator:
    """Generate different market scenarios for testing strategies.

    生成三种类型的测试场景:
    1. 合成 Hawkes 数据 - 不同聚类程度
    2. 真实历史数据 - 不同波动率期间
    3. 压力测试场景 - 极端市场条件
    """

    # 预定义的 Hawkes 参数组合
    HAWKES_SCENARIOS = {
        "low_clustering": HawkesScenarioConfig(
            name="low_clustering",
            mu=0.1, alpha=0.1, beta=0.8,
            T=86400.0
        ),
        "medium_clustering": HawkesScenarioConfig(
            name="medium_clustering",
            mu=0.1, alpha=0.4, beta=0.8,
            T=86400.0
        ),
        "high_clustering": HawkesScenarioConfig(
            name="high_clustering",
            mu=0.1, alpha=0.7, beta=0.8,
            T=86400.0
        ),
        "critical": HawkesScenarioConfig(
            name="critical",
            mu=0.1, alpha=0.9, beta=1.0,
            T=86400.0
        ),
    }

    def __init__(self, base_price: float = 50000.0, price_volatility: float = 0.02):
        """Initialize scenario generator.

        Args:
            base_price: Base price for synthetic data generation
            price_volatility: Base volatility for price simulation
        """
        self.base_price = base_price
        self.price_volatility = price_volatility

    def generate_hawkes_scenarios(self, seed_offset: int = 0) -> Dict[str, pd.DataFrame]:
        """Generate synthetic market data with different Hawkes clustering levels.

        生成不同聚类程度的合成数据:
        - 低聚类 (α=0.1, β=0.8): 接近泊松过程
        - 中聚类 (α=0.4, β=0.8): 典型市场条件
        - 高聚类 (α=0.7, β=0.8): 高活跃期
        - 临界状态 (α=0.9, β=1.0): 接近不稳定

        Args:
            seed_offset: Offset for random seeds to ensure different runs

        Returns:
            Dictionary mapping scenario names to DataFrames
        """
        scenarios = {}

        for name, config in self.HAWKES_SCENARIOS.items():
            logger.info(f"Generating Hawkes scenario: {name}",
                       extra=log_extra(
                           alpha=config.alpha,
                           beta=config.beta,
                           branching_ratio=config.branching_ratio
                       ))

            df = self._generate_single_hawkes_scenario(config, seed_offset)
            scenarios[name] = df

        return scenarios

    def _generate_single_hawkes_scenario(
        self,
        config: HawkesScenarioConfig,
        seed_offset: int
    ) -> pd.DataFrame:
        """Generate a single Hawkes scenario as market data."""
        params = HawkesParameters(mu=config.mu, alpha=config.alpha, beta=config.beta)
        process = HawkesProcess(params)

        # Simulate event times
        events = process.simulate(config.T, seed=config.seed + seed_offset)

        # Generate price data based on events
        return self._events_to_market_data(events, config)

    def _events_to_market_data(
        self,
        events: List[float],
        config: HawkesScenarioConfig
    ) -> pd.DataFrame:
        """Convert Hawkes events to market DataFrame with price and volume."""
        if not events:
            return pd.DataFrame()

        # Create timestamps from event times
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(seconds=t) for t in events]

        # Generate price path using random walk with volatility proportional to intensity
        prices = []
        volumes = []
        current_price = self.base_price

        for i, t in enumerate(events):
            # Intensity at this event affects volatility
            if i > 0:
                dt = t - events[i-1]
                intensity = config.mu + config.alpha * np.exp(-config.beta * dt)
            else:
                intensity = config.mu

            # Higher intensity = higher volatility
            local_vol = self.price_volatility * (1 + intensity)
            price_change = np.random.normal(0, local_vol * current_price / 100)
            current_price += price_change
            prices.append(current_price)

            # Volume proportional to intensity
            volume = max(0.1, np.random.exponential(intensity * 10))
            volumes.append(volume)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'intensity': [config.mu] + [
                config.mu + config.alpha * np.exp(-config.beta * (events[i] - events[i-1]))
                for i in range(1, len(events))
            ]
        })
        df.set_index('timestamp', inplace=True)

        return df

    def load_real_scenarios(
        self,
        date_ranges: List[Tuple[datetime, datetime]],
        data_dir: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load real historical data for different periods.

        Args:
            date_ranges: List of (start, end) datetime tuples
            data_dir: Directory containing historical data files

        Returns:
            Dictionary mapping period names to DataFrames
        """
        scenarios = {}

        data_dir = data_dir or os.getenv("DERIBIT_OPTIONS_DATA_DIR", "./data")

        for start, end in date_ranges:
            period_name = f"{start.strftime('%Y%m')}_{end.strftime('%Y%m')}"

            try:
                df = self._load_period_data(start, end, data_dir)
                if not df.empty:
                    scenarios[period_name] = df
                    logger.info(f"Loaded real data for period: {period_name}")
            except Exception as e:
                logger.warning(f"Failed to load data for {period_name}: {e}")

        return scenarios

    def _load_period_data(
        self,
        start: datetime,
        end: datetime,
        data_dir: str
    ) -> pd.DataFrame:
        """Load data for a specific period."""
        # This is a placeholder - actual implementation would load from files
        # For now, generate synthetic data with realistic characteristics
        logger.info(f"Loading/Generating data for {start} to {end}")

        # Generate synthetic data as fallback
        days = (end - start).days
        config = HawkesScenarioConfig(
            name="real_proxy",
            mu=0.1,
            alpha=0.4,
            beta=0.8,
            T=days * 86400.0
        )

        return self._generate_single_hawkes_scenario(config, seed_offset=start.month)

    def generate_stress_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Generate stress test scenarios.

        压力测试场景:
        - 突发交易量: 模拟新闻事件
        - 流动性枯竭: 宽价差、少成交
        - 库存极限: 持续单向成交
        """
        scenarios = {}

        # Scenario 1: Volume spike (news event)
        scenarios["volume_spike"] = self._generate_volume_spike_scenario()

        # Scenario 2: Liquidity drought
        scenarios["liquidity_drought"] = self._generate_liquidity_drought_scenario()

        # Scenario 3: Inventory stress (one-sided fills)
        scenarios["inventory_stress"] = self._generate_inventory_stress_scenario()

        return scenarios

    def _generate_volume_spike_scenario(self) -> pd.DataFrame:
        """Generate scenario with sudden volume spike mid-day."""
        config = HawkesScenarioConfig(
            name="volume_spike",
            mu=0.1,
            alpha=0.4,
            beta=0.8,
            T=86400.0
        )

        # Generate base events
        params = HawkesParameters(mu=config.mu, alpha=config.alpha, beta=config.beta)
        process = HawkesProcess(params)
        events = process.simulate(config.T, seed=42)

        # Add volume spike in the middle
        spike_start = 43200  # Noon
        spike_duration = 3600  # 1 hour
        # Keep alpha < beta for stable Hawkes process.
        spike_params = HawkesParameters(mu=0.5, alpha=0.8, beta=1.5)
        spike_process = HawkesProcess(spike_params)
        spike_events = [e + spike_start for e in spike_process.simulate(spike_duration, seed=43)]

        # Combine and sort
        all_events = sorted(events + spike_events)

        return self._events_to_market_data(all_events, config)

    def _generate_liquidity_drought_scenario(self) -> pd.DataFrame:
        """Generate scenario with reduced trading activity."""
        # Low baseline intensity with occasional bursts
        config = HawkesScenarioConfig(
            name="liquidity_drought",
            mu=0.02,  # Very low baseline
            alpha=0.3,
            beta=0.8,
            T=86400.0
        )

        params = HawkesParameters(mu=config.mu, alpha=config.alpha, beta=config.beta)
        process = HawkesProcess(params)
        events = process.simulate(config.T, seed=44)

        return self._events_to_market_data(events, config)

    def _generate_inventory_stress_scenario(self) -> pd.DataFrame:
        """Generate scenario with persistent one-sided order flow."""
        config = HawkesScenarioConfig(
            name="inventory_stress",
            mu=0.1,
            alpha=0.4,
            beta=0.8,
            T=86400.0
        )

        params = HawkesParameters(mu=config.mu, alpha=config.alpha, beta=config.beta)
        process = HawkesProcess(params)
        events = process.simulate(config.T, seed=45)

        # Generate market data
        df = self._events_to_market_data(events, config)

        # Add directional bias to simulate one-sided flow
        df['price'] = df['price'] * (1 + np.linspace(0, 0.05, len(df)))
        df['flow_direction'] = 1  # Persistent buy pressure

        return df


@dataclass
class HawkesSpecificMetrics:
    """Hawkes strategy specific metrics."""
    avg_intensity: float
    intensity_volatility: float
    intensity_spread_correlation: float
    adverse_selection_accuracy: float
    parameter_stability: Optional[pd.DataFrame] = None


class HawkesMetricsCollector:
    """Collect Hawkes-specific metrics during backtesting.

    收集 Hawkes 策略特有的指标:
    - 平均 Hawkes 强度 λ(t)
    - 价差与强度相关系数
    - 参数稳定性（Adaptive 版本）
    - 逆向选择检测准确率
    """

    def __init__(self):
        self.intensity_history: List[Tuple[datetime, float]] = []
        self.spread_history: List[Tuple[datetime, float]] = []
        self.parameter_history: List[Dict[str, Any]] = []
        self.adverse_selection_signals: List[Dict[str, Any]] = []

    def record_intensity(self, timestamp: datetime, intensity: float):
        """Record Hawkes intensity at a point in time."""
        self.intensity_history.append((timestamp, intensity))

    def record_spread(self, timestamp: datetime, spread_bps: float):
        """Record quoted spread at a point in time."""
        self.spread_history.append((timestamp, spread_bps))

    def record_parameters(
        self,
        timestamp: datetime,
        mu: float,
        alpha: float,
        beta: float
    ):
        """Record Hawkes parameter estimates (for adaptive strategies)."""
        self.parameter_history.append({
            'timestamp': timestamp,
            'mu': mu,
            'alpha': alpha,
            'beta': beta,
            'branching_ratio': alpha / beta if beta > 0 else 0
        })

    def record_adverse_selection_signal(
        self,
        timestamp: datetime,
        detected: bool,
        actual_adverse: bool
    ):
        """Record adverse selection detection result."""
        self.adverse_selection_signals.append({
            'timestamp': timestamp,
            'detected': detected,
            'actual_adverse': actual_adverse,
            'correct': detected == actual_adverse
        })

    def compute_metrics(self) -> HawkesSpecificMetrics:
        """Compute all Hawkes-specific metrics."""
        # Average intensity
        intensities = [i[1] for i in self.intensity_history]
        avg_intensity = np.mean(intensities) if intensities else 0.0
        intensity_vol = np.std(intensities) if intensities else 0.0

        # Intensity-spread correlation
        corr = self._compute_intensity_spread_correlation()

        # Adverse selection accuracy
        acc = self._compute_adverse_selection_accuracy()

        # Parameter stability
        param_stability = None
        if self.parameter_history:
            param_stability = pd.DataFrame(self.parameter_history)
            param_stability.set_index('timestamp', inplace=True)

        return HawkesSpecificMetrics(
            avg_intensity=avg_intensity,
            intensity_volatility=intensity_vol,
            intensity_spread_correlation=corr,
            adverse_selection_accuracy=acc,
            parameter_stability=param_stability
        )

    def _compute_intensity_spread_correlation(self) -> float:
        """Compute correlation between intensity and spread.

        理论上，高强度应该对应窄价差（更激进的报价）。
        因此相关系数应该为负值。
        """
        if not self.intensity_history or not self.spread_history:
            return 0.0

        # Align timestamps
        intensity_df = pd.DataFrame(self.intensity_history, columns=['timestamp', 'intensity'])
        spread_df = pd.DataFrame(self.spread_history, columns=['timestamp', 'spread'])

        intensity_df.set_index('timestamp', inplace=True)
        spread_df.set_index('timestamp', inplace=True)

        # Merge and compute correlation
        merged = pd.merge_asof(
            intensity_df.sort_index(),
            spread_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest'
        )

        if len(merged) > 1:
            return merged['intensity'].corr(merged['spread'])
        return 0.0

    def _compute_adverse_selection_accuracy(self) -> float:
        """Compute accuracy of adverse selection detection."""
        if not self.adverse_selection_signals:
            return 0.0

        correct = sum(1 for s in self.adverse_selection_signals if s['correct'])
        return correct / len(self.adverse_selection_signals)

    def get_intensity_series(self) -> pd.Series:
        """Get intensity as pandas Series."""
        if not self.intensity_history:
            return pd.Series()
        return pd.Series(
            [i[1] for i in self.intensity_history],
            index=[i[0] for i in self.intensity_history]
        )


@dataclass
class ComparisonResult:
    """Complete comparison result for a single scenario."""
    scenario_name: str
    scenario_type: ScenarioType
    scorecards: Dict[str, StrategyScorecard]
    hawkes_metrics: Optional[Dict[str, HawkesSpecificMetrics]] = None
    comparison_df: Optional[pd.DataFrame] = None


class ComprehensiveHawkesComparison:
    """Comprehensive comparison framework for Hawkes strategies.

    综合对比框架，运行所有场景并生成完整报告。
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 2.0
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.scenario_generator = ScenarioGenerator()
        self.results: Dict[str, ComparisonResult] = {}

    def run_full_comparison(
        self,
        strategies: List[MarketMakingStrategy],
        scenarios: Optional[Dict[str, pd.DataFrame]] = None,
        verbose: bool = True
    ) -> Dict[str, ComparisonResult]:
        """Run complete comparison across all scenarios.

        Args:
            strategies: List of strategies to compare
            scenarios: Optional pre-generated scenarios. If None, generates default set.
            verbose: Whether to print progress

        Returns:
            Dictionary of scenario names to ComparisonResults
        """
        if scenarios is None:
            scenarios = self._generate_default_scenarios()

        self.results = {}

        for scenario_name, market_data in scenarios.items():
            if verbose:
                logger.info(f"Running comparison for scenario: {scenario_name}")

            result = self._run_single_scenario(
                scenario_name,
                market_data,
                strategies,
                verbose
            )
            self.results[scenario_name] = result

        return self.results

    def _generate_default_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Generate default set of test scenarios."""
        scenarios = {}

        # Add Hawkes scenarios
        scenarios.update(self.scenario_generator.generate_hawkes_scenarios())

        # Add stress scenarios
        scenarios.update(self.scenario_generator.generate_stress_scenarios())

        return scenarios

    def _run_single_scenario(
        self,
        scenario_name: str,
        market_data: pd.DataFrame,
        strategies: List[MarketMakingStrategy],
        verbose: bool
    ) -> ComparisonResult:
        """Run comparison for a single scenario."""
        arena = StrategyArena(
            market_data=market_data,
            initial_capital=self.initial_capital,
            transaction_cost_bps=self.transaction_cost_bps
        )

        # Run tournament
        comparison_df = arena.run_tournament(strategies, verbose=verbose)

        # Collect Hawkes-specific metrics for Hawkes strategies
        hawkes_metrics = {}
        for strategy in strategies:
            if 'Hawkes' in strategy.name:
                metrics = self._extract_hawkes_metrics(strategy)
                if metrics:
                    hawkes_metrics[strategy.name] = metrics

        # Determine scenario type
        scenario_type = self._classify_scenario(scenario_name)

        return ComparisonResult(
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            scorecards=arena.scorecards,
            hawkes_metrics=hawkes_metrics if hawkes_metrics else None,
            comparison_df=comparison_df
        )

    def _extract_hawkes_metrics(
        self,
        strategy: MarketMakingStrategy
    ) -> Optional[HawkesSpecificMetrics]:
        """Extract Hawkes metrics from strategy if available."""
        # Check if strategy has a metrics collector
        if hasattr(strategy, 'metrics_collector'):
            return strategy.metrics_collector.compute_metrics()

        # Check if strategy has internal state with intensity history
        internal_state = strategy.get_internal_state()
        if 'intensity_history' in internal_state:
            collector = HawkesMetricsCollector()
            for entry in internal_state['intensity_history']:
                collector.record_intensity(
                    entry['timestamp'],
                    entry['intensity']
                )
            return collector.compute_metrics()

        return None

    def _classify_scenario(self, name: str) -> ScenarioType:
        """Classify scenario by name."""
        if 'clustering' in name or name in ['low', 'medium', 'high', 'critical']:
            return ScenarioType.SYNTHETIC_HAWKES
        elif 'spike' in name or 'drought' in name or 'stress' in name:
            return ScenarioType.STRESS_TEST
        else:
            return ScenarioType.REAL_HISTORICAL

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("HAWKES STRATEGY COMPREHENSIVE COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Scenarios Tested: {len(self.results)}")

        # Overall rankings
        lines.append("\n" + "=" * 80)
        lines.append("OVERALL STRATEGY RANKINGS")
        lines.append("=" * 80)

        for metric in ['sharpe_ratio', 'total_pnl', 'max_drawdown']:
            lines.append(f"\nBest by {metric}:")
            winners = self._aggregate_rankings(metric)
            for strategy, score in winners[:3]:
                lines.append(f"  {strategy}: {score:.4f}")

        # Scenario-specific results
        lines.append("\n" + "=" * 80)
        lines.append("SCENARIO-SPECIFIC RESULTS")
        lines.append("=" * 80)

        for name, result in self.results.items():
            lines.append(f"\n{name.upper()} ({result.scenario_type.value})")
            lines.append("-" * 40)

            if result.comparison_df is not None:
                for _, row in result.comparison_df.iterrows():
                    lines.append(
                        f"  {row['Strategy']}: "
                        f"PnL=${row['Total PnL ($)']:,.0f}, "
                        f"Sharpe={row['Sharpe']:.2f}"
                    )

        # Hawkes-specific insights
        lines.append("\n" + "=" * 80)
        lines.append("HAWKES-SPECIFIC INSIGHTS")
        lines.append("=" * 80)

        for name, result in self.results.items():
            if result.hawkes_metrics:
                lines.append(f"\n{name}:")
                for strat, metrics in result.hawkes_metrics.items():
                    lines.append(f"  {strat}:")
                    lines.append(f"    Avg Intensity: {metrics.avg_intensity:.4f}")
                    lines.append(f"    Intensity-Spread Corr: {metrics.intensity_spread_correlation:.4f}")
                    lines.append(f"    Adverse Selection Accuracy: {metrics.adverse_selection_accuracy:.2%}")

        return "\n".join(lines)

    def _aggregate_rankings(self, metric: str) -> List[Tuple[str, float]]:
        """Aggregate rankings across all scenarios."""
        scores = {}

        for result in self.results.values():
            for name, sc in result.scorecards.items():
                if name not in scores:
                    scores[name] = []
                scores[name].append(getattr(sc, metric, 0))

        # Average scores
        avg_scores = {
            name: np.mean(vals) if vals else 0
            for name, vals in scores.items()
        }

        # Sort (reverse for most metrics, but not for drawdown)
        reverse = metric != 'max_drawdown'
        return sorted(avg_scores.items(), key=lambda x: x[1], reverse=reverse)

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get aggregated comparison DataFrame across all scenarios."""
        rows = []

        for scenario_name, result in self.results.items():
            if result.comparison_df is not None:
                df = result.comparison_df.copy()
                df['Scenario'] = scenario_name
                df['ScenarioType'] = result.scenario_type.value
                rows.append(df)

        if rows:
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame()


class HawkesStrategyAnalyzer:
    """Analyze Hawkes strategy behavior in detail.

    提供详细的 Hawkes 策略行为分析，包括:
    - 强度-价差关系可视化数据
    - 参数收敛分析（自适应版本）
    - 逆向选择检测效果
    """

    def __init__(self, comparison_result: ComparisonResult):
        self.result = comparison_result

    def analyze_intensity_spread_relationship(self) -> Dict[str, Any]:
        """Analyze the relationship between intensity and spread.

        Returns data for scatter plot with regression line.
        """
        analysis = {}

        for strat_name, metrics in (self.result.hawkes_metrics or {}).items():
            if metrics.parameter_stability is not None:
                df = metrics.parameter_stability

                analysis[strat_name] = {
                    'correlation': metrics.intensity_spread_correlation,
                    'avg_intensity': metrics.avg_intensity,
                    'intensity_volatility': metrics.intensity_volatility
                }

        return analysis

    def analyze_parameter_convergence(self) -> Dict[str, pd.DataFrame]:
        """Analyze parameter stability over time."""
        convergence = {}

        for strat_name, metrics in (self.result.hawkes_metrics or {}).items():
            if metrics.parameter_stability is not None:
                df = metrics.parameter_stability

                # Compute rolling statistics
                if len(df) > 10:
                    convergence[strat_name] = {
                        'rolling_mean_alpha': df['alpha'].rolling(10).mean(),
                        'rolling_std_alpha': df['alpha'].rolling(10).std(),
                        'final_params': df.iloc[-1].to_dict() if len(df) > 0 else {}
                    }

        return convergence

    def get_adverse_selection_stats(self) -> Dict[str, Dict[str, float]]:
        """Get adverse selection detection statistics."""
        stats = {}

        for strat_name, metrics in (self.result.hawkes_metrics or {}).items():
            stats[strat_name] = {
                'accuracy': metrics.adverse_selection_accuracy,
                'avg_intensity': metrics.avg_intensity
            }

        return stats
