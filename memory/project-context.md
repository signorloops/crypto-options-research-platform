# CORP 项目上下文记录

> 生成时间: 2026-02-07
> 状态: 257/257 测试全部通过，代码覆盖率 72%
>
> ## 本次更新（性能优化）
> - **test_synthetic_data.py**: 优化 CompleteMarketSimulator 测试，从 109 秒降至 6.87 秒（16 倍提升）
> - **test_backtest.py**: 使用 session-scoped fixture 减少数据生成次数
> - **CompleteMarketSimulator.generate()**: 添加 `hours` 参数支持短时间模拟
> - **总测试时间**: 从 ~6 分钟降至 ~2 分钟（3 倍提升）
> - **pyproject.toml**: 修复 ruff 和 mypy 配置

---

## 项目概述

**名称**: CORP (Crypto Options Research Platform)
**用途**: 加密货币期权做市策略研究与回测
**位置**: `<PROJECT_ROOT>/corp`

---

## 已完成的工作

### 1. 代码审查修复
- WebSocket URL: `wbs://` → `wss://`
- 修复所有 `corp.` 前缀导入
- 修复 mid_price 0值处理
- 修复库存限制逻辑
- 修复 Greeks 输入验证
- 修复 pyproject.toml 包发现

### 2. 测试修复
- ✅ 原始 67 个测试全部通过
- ✅ `test_volatility.py` - 22 个波动率模块测试
- ✅ `test_arbitrage.py` - 29 个套利策略测试
- ✅ `test_validation.py` - 47 个数据验证测试
- ✅ `test_logging.py` - 15 个日志配置测试
- ✅ `test_microstructure.py` - 17 个微观结构分析测试
- ✅ `test_risk.py` - 23 个风险管理测试

---

## 项目结构

```
corp/
├── core/
│   ├── types.py               # OrderBook, Tick, Trade, Greeks
│   └── validation/            # ✅ Pydantic 数据验证
├── data/
├── data/
│   ├── downloaders/           # Deribit, Binance (含 WebSocket)
│   ├── generators/            # GBM, Merton Jump
│   ├── cache.py               # Parquet 缓存
│   └── streaming.py           # WebSocket 实时流 ✅
├── strategies/
│   ├── market_making/         # 4种做市策略
│   └── arbitrage/             # ✅ 4种套利策略
├── research/
│   ├── backtest/engine.py     # 回测引擎
│   ├── microstructure/        # VPIN, 订单簿特征
│   ├── risk/                  # Greeks, VaR
│   └── volatility/            # ✅ 波动率模型
├── utils/                     # ✅ 日志配置
├── tests/                     # 165个测试 ✅
├── .env.example               # ✅ 环境变量模板
└── notebooks/                 # Jupyter notebooks
```

---

## 待办事项

### 高优先级 ✅ 已完成
- [x] WebSocket 实时流 (`subscribe_order_book`, `subscribe_trades`)
- [x] 创建 `.env.example`
- [x] 添加日志配置 (`utils/logging_config.py`)

### 中优先级 ✅ 已完成
- [x] 完成 `research/volatility/` 模块 (22 个测试)
  - historical.py - Realized, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
  - models.py - EWMA, GARCH, HAR-RV
  - implied.py - Black-Scholes, IV calculation, VolatilitySurface
- [x] 完成 `strategies/arbitrage/` 模块 (29 个测试)
  - cross_exchange.py - 跨交易所套利
  - basis.py - 期现套利
  - option_box.py - 期权盒式套利
  - conversion.py - 转换/反转套利
- [x] 数据验证集成 (47 个测试)
  - core/validation/schemas.py - Pydantic 模型验证
  - TickData, TradeData, OrderBookData, GreeksData 等
  - BacktestConfig, DownloadRequest, WebSocketConfig 配置验证

### 低优先级 ✅ 已完成
- [x] 文档完善
  - docs/architecture.md - 架构图和模块详解
  - docs/quickstart.md - 快速开始指南
  - docs/api.md - 完整 API 文档
  - docs/deployment.md - 部署指南
  - README.md - 更新项目介绍
- [x] 代码覆盖率提升 (42% → 72%，257 个测试)
- [x] 测试性能优化 (总时间从 6 分钟降至 2 分钟)

---

## 代码覆盖率报告

> 更新时间: 2026-02-07

| 模块 | 覆盖率 | 状态 |
|------|--------|------|
| `core/types.py` | 98% | ✅ |
| `core/validation/schemas.py` | 96% | ✅ |
| `research/risk/greeks.py` | 97% | ✅ |
| `research/microstructure/orderbook_features.py` | 94% | ✅ |
| `utils/logging_config.py` | 92% | ✅ |
| `strategies/arbitrage/basis.py` | 92% | ✅ |
| `strategies/arbitrage/cross_exchange.py` | 92% | ✅ |
| `research/microstructure/vpin.py` | 84% | ✅ |
| `research/backtest/engine.py` | 82% | ✅ |
| `research/risk/var.py` | 89% | ✅ |
| **总计** | **72%** | ✅ |

### 覆盖率提升历史
- **初始**: 42% (67 个测试)
- **第一次提升**: 52% (新增 98 个测试)
- **第二次提升**: 68% (220 个测试, +55 个新测试)
- **当前**: 72% (257 个测试, 含性能优化)

---

## API Key 需求

### 当前: 无需 API Key
- Deribit: 公共 API 足够
- Binance: 公共 API 足够

### 可选 (更高频率限制)
```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

---

## 启动命令

```bash
cd "<PROJECT_ROOT>/corp"
source venv/bin/activate
python -m pytest tests/ -v
```

---

## 最近修改的文件

### 本次新增
1. `.env.example` - 环境变量模板
2. `utils/logging_config.py` - 统一日志配置
3. `utils/__init__.py` - 工具模块导出
4. `data/downloaders/deribit.py` - WebSocket 订阅集成
5. `data/downloaders/binance.py` - WebSocket 订阅集成
6. `data/__init__.py` - 模块导出

### Volatility 模块
7. `research/volatility/__init__.py` - 模块初始化
8. `research/volatility/historical.py` - 历史波动率计算
9. `research/volatility/models.py` - 波动率预测模型
10. `research/volatility/implied.py` - 隐含波动率计算
11. `tests/test_volatility.py` - 波动率模块测试 (22个)

### Arbitrage 模块
12. `strategies/arbitrage/__init__.py` - 模块初始化
13. `strategies/arbitrage/cross_exchange.py` - 跨交易所套利
14. `strategies/arbitrage/basis.py` - 期现套利
15. `strategies/arbitrage/option_box.py` - 期权盒式套利
16. `strategies/arbitrage/conversion.py` - 转换套利
17. `tests/test_arbitrage.py` - 套利策略测试 (29个)

### 数据验证模块
18. `core/validation/__init__.py` - 验证模块初始化
19. `core/validation/schemas.py` - Pydantic 验证模型
20. `core/validation/validators.py` - 自定义验证函数
21. `tests/test_validation.py` - 数据验证测试 (47个)

### 新增测试模块（覆盖率提升）
22. `tests/test_logging.py` - 日志配置测试 (15个)
23. `tests/test_microstructure.py` - 微观结构分析测试 (17个)
24. `tests/test_risk.py` - 风险管理测试 (23个)
25. `utils/logging_config.py` - 添加 `force` 参数支持测试
26. `research/microstructure/vpin.py` - 修复 datetime 类型问题

---

*下次启动时读取此文件恢复上下文*
