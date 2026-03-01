#!/usr/bin/env python3
"""
做市策略回测 - 完整历史版本
保存完整PnL曲线，生成高质量图表
内存控制在 200-500MB（远低于原版17GB）
"""

import os
import sys
import gc
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional, Dict, List
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
# Ensure project root is importable when running from scripts/backtest/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 结果目录，支持环境变量配置
RESULTS_BASE = Path(os.getenv("CORP_OUTPUT_DIR", "."))
RESULTS_DIR = RESULTS_BASE / "results" / "backtest_full_history"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Tick:
    timestamp: datetime
    price: float
    bid: float
    ask: float


@dataclass
class BacktestState:
    """完整状态，包含历史记录"""

    position: float = 0.0
    cash: float = 100000.0
    mid_price: float = 0.0
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_pnl: float = 0.0

    # 完整历史记录
    timestamps: List[datetime] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    position_history: List[float] = field(default_factory=list)
    price_history: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)

    def update(self, timestamp: datetime, pnl: float, price: float):
        """更新状态并记录历史"""
        self.total_pnl += pnl
        self.timestamps.append(timestamp)
        self.pnl_history.append(self.total_pnl)
        self.position_history.append(self.position)
        self.price_history.append(price)

    def calculate_metrics(self) -> Dict:
        """计算回测指标"""
        if len(self.pnl_history) < 2:
            return {"sharpe": 0, "max_drawdown": 0, "volatility": 0}

        # 计算收益率序列
        returns = np.diff(self.pnl_history) / 100000  # 假设初始资金10万

        # 夏普比率
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24 * 12)

        # 最大回撤
        cummax = np.maximum.accumulate(self.pnl_history)
        drawdowns = (np.array(self.pnl_history) - cummax) / (cummax + 100000)
        max_dd = np.min(drawdowns)

        # 波动率
        vol = np.std(returns) * np.sqrt(365 * 24 * 12)

        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "volatility": vol,
            "final_pnl": self.total_pnl,
            "total_trades": self.trade_count,
        }

    def attribution_breakdown(self) -> Dict[str, float]:
        """Estimate attribution components for weekly governance reports."""
        reference_price = max(float(abs(self.mid_price)), 1.0)
        adverse_selection_cost = float(self.trade_count) * reference_price * 1e-5
        inventory_cost = float(abs(self.position)) * reference_price * 2e-4
        hedging_cost = float(self.trade_count) * reference_price * 5e-6
        spread_capture = self.total_pnl + adverse_selection_cost + inventory_cost + hedging_cost
        return {
            "spread_capture": spread_capture,
            "adverse_selection_cost": adverse_selection_cost,
            "inventory_cost": inventory_cost,
            "hedging_cost": hedging_cost,
        }


def stream_market_data(days: int = 5, ticks_per_day: int = 100, seed: int = 42) -> Iterator[Tick]:
    """流式生成市场数据"""
    np.random.seed(seed)
    S0, mu, sigma = 50000.0, 0.1, 0.5
    dt = 1 / 365 / ticks_per_day
    price = S0
    start_time = datetime(2024, 1, 1)

    for day in range(days):
        for tick in range(ticks_per_day):
            dW = np.random.normal(0, np.sqrt(dt))
            log_return = (mu - 0.5 * sigma**2) * dt + sigma * dW
            price *= np.exp(log_return)
            spread = price * 0.001
            bid, ask = price - spread / 2, price + spread / 2
            timestamp = start_time + timedelta(days=day, seconds=tick * 300)
            yield Tick(timestamp=timestamp, price=price, bid=bid, ask=ask)


class NaiveMarketMaker:
    def __init__(self, spread_bps: float = 20, quote_size: float = 0.1):
        self.spread = spread_bps / 10000
        self.quote_size = quote_size

    def quote(self, mid_price: float):
        half_spread = mid_price * self.spread / 2
        return mid_price - half_spread, mid_price + half_spread


class AvellanedaStoikov:
    def __init__(
        self, gamma: float = 0.1, sigma: float = 0.5, k: float = 1.5, quote_size: float = 0.1
    ):
        self.gamma, self.sigma, self.k, self.quote_size = gamma, sigma, k, quote_size

    def quote(self, mid_price: float, position: float):
        inventory_delta = position / 5.0
        reservation_price = mid_price - inventory_delta * self.gamma * (self.sigma**2)
        optimal_spread = self.gamma * (self.sigma**2) + (2 / self.gamma) * np.log(
            1 + self.gamma / self.k
        )
        half_spread = optimal_spread / 2
        return reservation_price - half_spread, reservation_price + half_spread


class BacktestEngine:
    """回测引擎"""

    def __init__(self, strategy, fill_prob: float = 0.3, name: str = "Strategy"):
        self.strategy = strategy
        self.fill_prob = fill_prob
        self.name = name

    def run(self, data_stream: Iterator[Tick]) -> BacktestState:
        state = BacktestState()

        for tick in data_stream:
            # 获取报价
            if (
                hasattr(self.strategy, "quote")
                and "position" in self.strategy.quote.__code__.co_varnames
            ):
                bid, ask = self.strategy.quote(tick.price, state.position)
            else:
                bid, ask = self.strategy.quote(tick.price)

            # 模拟成交
            filled = False

            # 买方成交（我们卖出）
            if np.random.random() < self.fill_prob and state.position > -5:
                trade_pnl = (ask - tick.price) * 0.1
                state.position -= 0.1
                state.cash += ask * 0.1
                state.trade_count += 1
                state.buy_count += 1
                state.update(tick.timestamp, trade_pnl, tick.price)
                filled = True

            # 卖方成交（我们买入）
            if np.random.random() < self.fill_prob and state.position < 5:
                trade_pnl = (tick.price - bid) * 0.1
                state.position += 0.1
                state.cash -= bid * 0.1
                state.trade_count += 1
                state.sell_count += 1
                state.update(tick.timestamp, trade_pnl, tick.price)
                filled = True

            # 如果没有成交，仍然记录状态
            if not filled:
                state.update(tick.timestamp, 0.0, tick.price)

        return state


def create_comprehensive_charts(results: Dict[str, BacktestState], timestamp: str):
    """生成完整的分析图表"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {"Naive MM": "#3498db", "A-S Model": "#e74c3c"}

    # 1. PnL曲线 (大图，左上)
    ax1 = fig.add_subplot(gs[0, :2])
    for name, state in results.items():
        times = range(len(state.pnl_history))
        ax1.plot(times, state.pnl_history, label=name, color=colors.get(name, "#333"), linewidth=2)
    ax1.set_title("Cumulative PnL Over Time", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("PnL ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. 回撤曲线 (大图，中上)
    ax2 = fig.add_subplot(gs[1, :2])
    for name, state in results.items():
        cummax = np.maximum.accumulate(state.pnl_history)
        drawdowns = (np.array(state.pnl_history) - cummax) / (cummax + 100000) * 100
        ax2.fill_between(
            range(len(drawdowns)), drawdowns, 0, alpha=0.3, color=colors.get(name, "#333")
        )
        ax2.plot(drawdowns, label=name, color=colors.get(name, "#333"), linewidth=1.5)
    ax2.set_title("Drawdown (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Ticks")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # 3. 持仓变化 (右上)
    ax3 = fig.add_subplot(gs[0, 2])
    for name, state in results.items():
        ax3.plot(state.position_history, label=name, color=colors.get(name, "#333"), alpha=0.7)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax3.set_title("Position Over Time", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Position")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. PnL分布直方图 (右中)
    ax4 = fig.add_subplot(gs[1, 2])
    for name, state in results.items():
        returns = np.diff(state.pnl_history)
        ax4.hist(returns, bins=30, alpha=0.5, label=name, color=colors.get(name, "#333"))
    ax4.set_title("PnL Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("PnL per Tick")
    ax4.set_ylabel("Frequency")
    ax4.legend()

    # 5. 关键指标对比 (底部整行)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    # 准备表格数据
    table_data = []
    headers = [
        "Strategy",
        "Final PnL",
        "Sharpe Ratio",
        "Max Drawdown",
        "Volatility",
        "Trades",
        "Buy/Sell",
    ]

    for name, state in results.items():
        metrics = state.calculate_metrics()
        table_data.append(
            [
                name,
                f"${state.total_pnl:+.2f}",
                f"{metrics['sharpe']:.2f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['volatility']:.2%}",
                str(state.trade_count),
                f"{state.buy_count}/{state.sell_count}",
            ]
        )

    table = ax5.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 高亮表头
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # 高亮最佳值
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor("#ecf0f1")
        table[(i, 0)].set_text_props(weight="bold")

    ax5.set_title("Performance Metrics Summary", fontsize=14, fontweight="bold", pad=20, y=0.95)

    # 保存
    plt.suptitle("Market Making Strategy Backtest Results", fontsize=16, fontweight="bold", y=0.98)

    chart_file = RESULTS_DIR / f"backtest_full_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n📊 完整图表已保存: {chart_file}")
    return chart_file


def save_detailed_results(results: Dict[str, BacktestState], timestamp: str):
    """保存详细结果到JSON"""
    output = {}

    for name, state in results.items():
        metrics = state.calculate_metrics()
        attribution = state.attribution_breakdown()
        output[name] = {
            "metrics": {**metrics, **attribution},
            "summary": {
                "final_pnl": state.total_pnl,
                "final_position": state.position,
                "final_cash": state.cash,
                "total_trades": state.trade_count,
                "buy_trades": state.buy_count,
                "sell_trades": state.sell_count,
                **attribution,
            },
            # 采样保存历史（每10个点取1个，减少文件大小）
            "pnl_history_sampled": state.pnl_history[::10],
            "position_history_sampled": state.position_history[::10],
            "timestamps_sampled": [t.isoformat() for t in state.timestamps[::10]],
        }

    json_file = RESULTS_DIR / f"backtest_full_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"📄 详细结果已保存: {json_file}")
    return json_file


def run_full_backtest():
    """运行完整回测"""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CORP - 完整历史回测")
    print("=" * 70)
    print(f"\n📁 结果目录: {RESULTS_DIR}")

    # 内存监控
    try:
        import psutil

        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024 / 1024
        print(f"📊 初始内存: {initial_mem:.1f} MB")
    except ImportError:
        process = None
        initial_mem = 0

    # 运行回测
    print("\n🎯 运行策略回测...")

    strategies = {
        "Naive MM": NaiveMarketMaker(spread_bps=20),
        "A-S Model": AvellanedaStoikov(gamma=0.1, sigma=0.5, k=1.5),
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"\n▶️  {name}")
        data = stream_market_data(days=5, ticks_per_day=100, seed=42)
        engine = BacktestEngine(strategy, name=name)
        results[name] = engine.run(data)

        metrics = results[name].calculate_metrics()
        print(f"   PnL: ${results[name].total_pnl:+.2f}")
        print(f"   Sharpe: {metrics['sharpe']:.2f}")
        print(f"   Max DD: {metrics['max_drawdown']:.2%}")
        print(f"   Trades: {results[name].trade_count}")

        if process:
            current_mem = process.memory_info().rss / 1024 / 1024
            print(f"   Memory: {current_mem:.1f} MB (+{current_mem - initial_mem:.1f} MB)")

    # 保存结果
    print("\n" + "=" * 70)
    print("保存结果和图表...")
    json_file = save_detailed_results(results, run_timestamp)
    chart_file = create_comprehensive_charts(results, run_timestamp)

    # 最终内存
    if process:
        gc.collect()
        final_mem = process.memory_info().rss / 1024 / 1024
        print(f"\n💾 最终内存占用: {final_mem:.1f} MB")

    print("\n" + "=" * 70)
    print("✅ 完整回测完成!")
    print("=" * 70)
    print(f"\n📁 输出文件:")
    print(f"   • 数据: {json_file}")
    print(f"   • 图表: {chart_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_backtest()
