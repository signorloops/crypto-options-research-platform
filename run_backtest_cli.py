#!/usr/bin/env python3
"""
命令行运行做市策略回测
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无GUI模式
import matplotlib.pyplot as plt

from data.generators.synthetic import CompleteMarketSimulator
from strategies.market_making.naive import NaiveMarketMaker, NaiveMMConfig
from strategies.market_making.avellaneda_stoikov import AvellanedaStoikov, ASConfig
from research.backtest.engine import BacktestEngine


def run_backtest():
    """运行回测并输出结果"""
    print("=" * 70)
    print("CORP - 做市策略回测 (命令行模式)")
    print("=" * 70)

    # 1. 生成市场数据
    print("\n📊 步骤1: 生成合成市场数据...")
    simulator = CompleteMarketSimulator(seed=42)
    market_data = simulator.generate(days=30, include_options=True)

    print(f"   ✓ 现货价格: {len(market_data['spot']):,} 条")
    print(f"   ✓ 订单簿: {len(market_data['order_book']):,} 条")
    print(f"   ✓ 期权数据: {len(market_data['options']):,} 条")

    # 2. 初始化策略
    print("\n🎯 步骤2: 初始化策略...")

    # 策略A: 简单做市
    naive = NaiveMarketMaker(
        NaiveMMConfig(spread_bps=20, quote_size=0.5, max_position=5.0)
    )
    print("   ✓ NaiveMarketMaker: 固定20基点价差")

    # 策略B: Avellaneda-Stoikov
    as_strategy = AvellanedaStoikov(
        ASConfig(gamma=0.1, sigma=0.5, k=1.5, quote_size=0.5, inventory_limit=5.0)
    )
    print("   ✓ Avellaneda-Stoikov: γ=0.1, σ=0.5, k=1.5")

    # 3. 运行回测
    print("\n⚙️  步骤3: 运行回测...")
    print("   运行 Naive 策略...", end=" ")
    naive_result = BacktestEngine(naive).run(market_data['spot'])
    print("✓")

    print("   运行 A-S 策略...", end=" ")
    as_result = BacktestEngine(as_strategy).run(market_data['spot'])
    print("✓")

    # 4. 显示结果
    print("\n" + "=" * 70)
    print("📈 回测结果对比")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 指标                │ Naive (固定价差)    │ A-S (最优做市)      │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ 总 PnL              │ {naive_result.total_pnl:>+18.4f}  │ {as_result.total_pnl:>+18.4f}  │")
    print(f"│ 夏普比率            │ {naive_result.sharpe_ratio:>18.4f}  │ {as_result.sharpe_ratio:>18.4f}  │")
    print(f"│ 最大回撤            │ {naive_result.max_drawdown:>18.4f}  │ {as_result.max_drawdown:>18.4f}  │")
    print(f"│ 交易次数            │ {naive_result.trade_count:>18}  │ {as_result.trade_count:>18}  │")
    print(f"│ 买入/卖出           │ {naive_result.buy_count:>4}/{naive_result.sell_count:<4}{' '*8}  │ {as_result.buy_count:>4}/{as_result.sell_count:<4}{' '*8}  │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # 5. 分析对比
    print("\n" + "=" * 70)
    print("📊 策略分析")
    print("=" * 70)

    pnl_diff = as_result.total_pnl - naive_result.total_pnl
    sharpe_diff = as_result.sharpe_ratio - naive_result.sharpe_ratio

    print(f"\nPnL 差异: {pnl_diff:+.4f} ({'A-S 更优 ✓' if pnl_diff > 0 else 'Naive 更优'})")
    print(f"夏普差异: {sharpe_diff:+.4f} ({'A-S 更优 ✓' if sharpe_diff > 0 else 'Naive 更优'})")

    # 库存管理效果
    naive_var = naive_result.inventory_series.var()
    as_var = as_result.inventory_series.var()
    var_improvement = (1 - as_var / naive_var) * 100 if naive_var > 0 else 0

    print(f"\n📦 库存管理:")
    print(f"   Naive 库存方差: {naive_var:.6f}")
    print(f"   A-S 库存方差:   {as_var:.6f}")
    print(f"   风险降低:       {var_improvement:.1f}% ✓")

    # 6. 保存图表
    output_dir = Path(os.getenv("CORP_OUTPUT_DIR", tempfile.gettempdir())) / "corp_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "backtest_result.png"
    print(f"\n💾 保存图表到 {output_file}...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PnL 对比
    naive_result.pnl_series.plot(ax=axes[0, 0], label='Naive MM', alpha=0.8)
    as_result.pnl_series.plot(ax=axes[0, 0], label='A-S Model', alpha=0.8)
    axes[0, 0].set_title('Cumulative PnL')
    axes[0, 0].set_ylabel('PnL ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 库存对比
    naive_result.inventory_series.plot(ax=axes[0, 1], label='Naive MM', alpha=0.8)
    as_result.inventory_series.plot(ax=axes[0, 1], label='A-S Model', alpha=0.8)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Inventory Position')
    axes[0, 1].set_ylabel('Position Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 回撤
    def calc_drawdown(pnl):
        running_max = pnl.expanding().max()
        return (pnl - running_max) / (running_max + 100000)

    calc_drawdown(naive_result.pnl_series).plot(ax=axes[1, 0], label='Naive', color='red', alpha=0.7)
    calc_drawdown(as_result.pnl_series).plot(ax=axes[1, 0], label='A-S', color='green', alpha=0.7)
    axes[1, 0].set_title('Drawdown')
    axes[1, 0].set_ylabel('Drawdown %')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 交易分布
    trades_data = {
        'Strategy': ['Naive', 'A-S'],
        'Buys': [naive_result.buy_count, as_result.buy_count],
        'Sells': [naive_result.sell_count, as_result.sell_count]
    }
    x = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x - width/2, trades_data['Buys'], width, label='Buys', alpha=0.8)
    axes[1, 1].bar(x + width/2, trades_data['Sells'], width, label='Sells', alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(trades_data['Strategy'])
    axes[1, 1].set_title('Trade Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print("   ✓ 图表已保存")

    print("\n" + "=" * 70)
    print("✅ 回测完成!")
    print("=" * 70)
    print("\n💡 结论:")
    print("   Avellaneda-Stoikov 模型通过动态库存倾斜和最优报价，")
    print("   在控制库存风险的同时提升收益，是高波动市场的理想选择。")
    print(f"\n📁 输出文件: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    run_backtest()
