# CORP 任务清单

## 已完成

- [x] **修复 Notebook 01 零成交 Bug**
  - 根因：`engine.py:472` 合成交易价格偏移量 `mid * 0.0001`（1 bps）太小，策略报价价差远大于此
  - 修复：使用方向感知的价格生成，trade 价格穿过 bid/ask
  - 文件：`research/backtest/engine.py` 第 470-478 行
  - 验证：trade_count=44（修复前为 0）

- [x] **修复 Notebook cell-9 属性名错误**
  - `BacktestResult` 没有 `total_pnl` 属性，应为 `total_pnl_usd`
  - 文件：`notebooks/01_market_simulation_demo.ipynb` cell-9

- [x] **创建学习路径文档**
  - `learning/learning-path-strategy-development.md`
  - `learning/theory.md`
  - `learning/python-concepts-quickref.md`

## 待办

- [x] 修复 `test_hawkes_comparison.py` 和 `test_strategy_arena.py` 的 collection error
- [x] 修复 `arena.py` 的 Python 3.9 annotation 兼容性问题
- [x] 为 Notebook 01 增加可重复执行的验证脚本与图表产物输出
- [x] 手工重跑 `notebooks/01_market_simulation_demo.ipynb` 并刷新嵌入输出
