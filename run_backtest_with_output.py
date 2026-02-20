#!/usr/bin/env python3
"""
内存优化的做市策略回测 - 带文件输出和图表
结果保存到: $CORP_OUTPUT_DIR/results/ 目录 (默认: ./results/)
"""

import os
import sys
import gc
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional, Dict, List
from datetime import datetime, timedelta
from collections import deque
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI模式
import matplotlib.pyplot as plt

# 创建结果目录，支持环境变量配置
RESULTS_BASE = Path(os.getenv("CORP_OUTPUT_DIR", "."))
RESULTS_DIR = RESULTS_BASE / "results" / "backtest_with_output"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============ 数据类型 ============
@dataclass(slots=True)
class Tick:
    timestamp: datetime
    price: float
    bid: float
    ask: float


@dataclass(slots=True)
class BacktestState:
    """精简回测状态"""
    position: float = 0.0
    cash: float = 100000.0
    mid_price: float = 0.0
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_pnl: float = 0.0
    _pnl_sum: float = 0.0
    _pnl_sum_sq: float = 0.0
    _max_nav: float = 0.0
    _min_nav: float = float('inf')

    # 新增：保存关键时间点数据用于绘图
    pnl_history: List[tuple] = field(default_factory=list)
    position_history: List[tuple] = field(default_factory=list)

    def update_pnl(self, pnl: float, nav: float, timestamp: datetime = None):
        self.total_pnl += pnl
        self._pnl_sum += pnl
        self._pnl_sum_sq += pnl * pnl
        self._max_nav = max(self._max_nav, nav)
        self._min_nav = min(self._min_nav, nav)

        # 记录历史（每100个tick记录一次，控制内存）
        if timestamp and len(self.pnl_history) < 1000:
            if len(self.pnl_history) == 0 or len(self.pnl_history) % 10 == 0:
                self.pnl_history.append((timestamp.isoformat(), self.total_pnl))
                self.position_history.append((timestamp.isoformat(), self.position))

    @property
    def sharpe_ratio(self) -> float:
        if self.trade_count < 2:
            return 0.0
        mean = self._pnl_sum / self.trade_count
        var = (self._pnl_sum_sq / self.trade_count) - (mean ** 2)
        std = np.sqrt(max(var, 1e-10))
        return mean / std * np.sqrt(365) if std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        if self._max_nav <= 0:
            return 0.0
        return (self._min_nav - self._max_nav) / self._max_nav

    def to_dict(self) -> Dict:
        return {
            "position": self.position,
            "cash": self.cash,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
        }


# ============ 流式数据生成 ============
def stream_market_data(days: int = 5, ticks_per_day: int = 100, seed: int = 42) -> Iterator[Tick]:
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


# ============ 策略 ============
class NaiveMarketMaker:
    def __init__(self, spread_bps: float = 20, quote_size: float = 0.1):
        self.spread = spread_bps / 10000
        self.quote_size = quote_size

    def quote(self, state: BacktestState):
        mid = state.mid_price
        half_spread = mid * self.spread / 2
        return mid - half_spread, mid + half_spread


class AvellanedaStoikov:
    def __init__(self, gamma: float = 0.1, sigma: float = 0.5, k: float = 1.5, quote_size: float = 0.1):
        self.gamma, self.sigma, self.k, self.quote_size = gamma, sigma, k, quote_size

    def quote(self, state: BacktestState):
        mid = state.mid_price
        inventory_delta = state.position / 5.0
        reservation_price = mid - inventory_delta * self.gamma * (self.sigma ** 2)
        optimal_spread = self.gamma * (self.sigma ** 2) + (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        half_spread = optimal_spread / 2
        return reservation_price - half_spread, reservation_price + half_spread


# ============ 回测引擎 ============
class StreamingBacktest:
    def __init__(self, strategy, fill_prob: float = 0.3):
        self.strategy = strategy
        self.fill_prob = fill_prob

    def run(self, data_stream: Iterator[Tick]) -> BacktestState:
        state = BacktestState()
        tick_count = 0

        for tick in data_stream:
            tick_count += 1
            state.mid_price = tick.price
            bid, ask = self.strategy.quote(state)

            # 买方成交
            if np.random.random() < self.fill_prob and state.position > -5:
                trade_pnl = (ask - tick.price) * 0.1
                state.position -= 0.1
                state.cash += ask * 0.1
                nav = state.cash + state.position * tick.price
                state.update_pnl(trade_pnl, nav, tick.timestamp)
                state.trade_count += 1
                state.sell_count += 1

            # 卖方成交
            if np.random.random() < self.fill_prob and state.position < 5:
                trade_pnl = (tick.price - bid) * 0.1
                state.position += 0.1
                state.cash -= bid * 0.1
                nav = state.cash + state.position * tick.price
                state.update_pnl(trade_pnl, nav, tick.timestamp)
                state.trade_count += 1
                state.buy_count += 1

        return state


# ============ 保存结果 ============
def save_results(results: Dict[str, BacktestState], timestamp: str):
    """保存结果到JSON文件"""
    output = {}
    for name, state in results.items():
        output[name] = {
            "summary": state.to_dict(),
            "pnl_history": state.pnl_history,
            "position_history": state.position_history,
        }

    output_file = RESULTS_DIR / f"backtest_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 结果已保存: {output_file}")
    return output_file


def create_charts(results: Dict[str, BacktestState], timestamp: str):
    """生成图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'Naive MM': '#3498db', 'A-S Model': '#e74c3c'}

    for name, state in results.items():
        label = name
        color = colors.get(name, '#333333')

        # PnL曲线
        if state.pnl_history:
            times = [t[0][11:19] for t in state.pnl_history]  # 只取时间部分
            pnls = [t[1] for t in state.pnl_history]
            axes[0, 0].plot(pnls, label=label, color=color, linewidth=2)

        # 持仓曲线
        if state.position_history:
            positions = [t[1] for t in state.position_history]
            axes[0, 1].plot(positions, label=label, color=color, linewidth=2)

    # PnL图设置
    axes[0, 0].set_title('累计 PnL', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 持仓图设置
    axes[0, 1].set_title('持仓变化', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('持仓量')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 交易统计柱状图
    strategies = list(results.keys())
    buys = [results[s].buy_count for s in strategies]
    sells = [results[s].sell_count for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35
    axes[1, 0].bar(x - width/2, buys, width, label='买入', alpha=0.8, color='#2ecc71')
    axes[1, 0].bar(x + width/2, sells, width, label='卖出', alpha=0.8, color='#e74c3c')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategies)
    axes[1, 0].set_title('交易分布', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 关键指标对比表
    axes[1, 1].axis('off')
    table_data = []
    for name, state in results.items():
        table_data.append([
            name,
            f"{state.total_pnl:+.2f}",
            f"{state.sharpe_ratio:.2f}",
            f"{state.max_drawdown:.2%}",
            str(state.trade_count)
        ])

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['策略', '总PnL', '夏普', '最大回撤', '交易次数'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    axes[1, 1].set_title('回测结果汇总', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    chart_file = RESULTS_DIR / f"backtest_chart_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"💾 图表已保存: {chart_file}")
    return chart_file


# ============ 主程序 ============
def run_backtest_with_output():
    """运行回测并保存结果"""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CORP - 做市策略回测 (带输出)")
    print("=" * 70)
    print(f"\n📁 结果目录: {RESULTS_DIR}")

    # 内存监控
    try:
        import psutil
        process = psutil.Process()
        print(f"📊 初始内存: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    except ImportError:
        process = None
        print("⚠️  安装 psutil 获取内存统计: pip install psutil")

    # 运行回测
    print("\n🎯 初始化策略...")
    strategies = {
        "Naive MM": NaiveMarketMaker(spread_bps=20, quote_size=0.1),
        "A-S Model": AvellanedaStoikov(gamma=0.1, sigma=0.5, k=1.5, quote_size=0.1),
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"\n▶️  运行 {name}...")
        data = stream_market_data(days=5, ticks_per_day=100, seed=42)
        engine = StreamingBacktest(strategy)
        results[name] = engine.run(data)
        print(f"   ✓ PnL: {results[name].total_pnl:+.2f} | 交易: {results[name].trade_count}")

    # 保存结果
    print("\n" + "=" * 70)
    print("保存结果...")
    json_file = save_results(results, run_timestamp)
    chart_file = create_charts(results, run_timestamp)

    # 内存统计
    if process:
        gc.collect()
        final_mem = process.memory_info().rss / 1024 / 1024
        print(f"\n💾 内存占用: {final_mem:.1f} MB")

    # 打印结果表格
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    print(f"\n{'策略':<20} {'PnL':>12} {'夏普':>10} {'回撤':>10} {'交易':>8}")
    print("-" * 70)
    for name, state in results.items():
        print(f"{name:<20} {state.total_pnl:>+12.2f} {state.sharpe_ratio:>10.2f} "
              f"{state.max_drawdown:>10.2%} {state.trade_count:>8}")

    print("\n" + "=" * 70)
    print("✅ 回测完成!")
    print("=" * 70)
    print(f"\n📁 输出文件:")
    print(f"   • 数据: {json_file}")
    print(f"   • 图表: {chart_file}")
    print("=" * 70)


if __name__ == '__main__':
    run_backtest_with_output()
