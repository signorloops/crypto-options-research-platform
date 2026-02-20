#!/usr/bin/env python3
"""做市策略回测演示"""

import sys
sys.path.insert(0, '.')

from data.generators.synthetic import CompleteMarketSimulator
from strategies.market_making.naive import NaiveMarketMaker, NaiveMMConfig
from strategies.market_making.avellaneda_stoikov import AvellanedaStoikov, ASConfig
from research.backtest.engine import BacktestEngine

print('=' * 60)
print('CORP - 做市策略回测演示')
print('=' * 60)

# 生成数据
print('\n1. 生成市场数据...')
simulator = CompleteMarketSimulator(seed=42)
market_data = simulator.generate(days=30, include_options=True)
print(f'   现货价格: {len(market_data["spot"])} 条')
print(f'   订单簿: {len(market_data["order_book"])} 条')

# 策略
print('\n2. 初始化策略...')
naive = NaiveMarketMaker(NaiveMMConfig(spread_bps=20, quote_size=0.5))
as_strategy = AvellanedaStoikov(ASConfig(gamma=0.1, sigma=0.5, k=1.5, quote_size=0.5))

# 回测
print('\n3. 运行回测...')
naive_result = BacktestEngine(naive).run(market_data['spot'])
as_result = BacktestEngine(as_strategy).run(market_data['spot'])

print('\n' + '=' * 60)
print('回测结果')
print('=' * 60)
print(f'\nNaive 策略:     PnL={naive_result.total_pnl:+.4f}, 夏普={naive_result.sharpe_ratio:.4f}')
print(f'A-S 策略:       PnL={as_result.total_pnl:+.4f}, 夏普={as_result.sharpe_ratio:.4f}')

print('\n' + '=' * 60)
print('演示完成！')
print('=' * 60)
