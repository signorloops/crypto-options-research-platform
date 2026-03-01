# 快速开始

## 1. 安装

```bash
git clone https://github.com/signorloops/crypto-options-research-platform.git
cd crypto-options-research-platform

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"
# 可选完整栈
pip install -e ".[dev,full]"

cp .env.example .env
```

## 2. 最小验证

```bash
pytest -q -m "not integration"
make docs-link-check
```

## 3. 基础示例

### 波动率

```python
import numpy as np
from research.volatility.historical import realized_volatility

returns = np.random.normal(0, 0.02, 500)
vol = realized_volatility(returns, annualize=True, periods=365)
print(vol)
```

### 回测

```python
from research.backtest.hawkes_comparison import ScenarioGenerator

gen = ScenarioGenerator(base_price=50000.0)
scenarios = gen.generate_hawkes_scenarios()
print(list(scenarios.keys()))
```

## 4. 常用命令

```bash
# 测试
pytest -q -m "not integration"
RUN_INTEGRATION_TESTS=1 pytest -q -m "integration"

# 代码质量
black .
ruff check . --fix
mypy .

# 研究审计
make research-audit
make research-audit-compare
make research-audit-refresh-baseline

# 治理链路
make complexity-audit
make weekly-operating-audit
make weekly-close-gate
make live-deviation-snapshot
```

## 5. 常见问题

1. `429 Too Many Requests`：降低请求频率或配置 API Key。
2. WebSocket 中断：检查网络，客户端已有自动重连。
3. 内存不足：缩小回测窗口或减少数据规模。

## 6. 下一步

- 导航入口：`GUIDE.md`
- 路线图：`plans/2026-Q2-long-term-execution-roadmap.md`
- 周清单：`plans/weekly-operating-checklist.md`
- 治理手册：`governance-operations.md`
- 治理时间线摘要：`archive/reports/2026-Q1-governance-timeline-summary.md`
- 数学验证摘要：`archive/reports/2026-Q1-math-validation-summary.md`
- 历史计划摘要：`archive/plans/2026-Q1-archived-plans-summary.md`
- 历史学习摘要：`archive/learning/learning-materials-summary.md`
- 历史报告摘要：`archive/reports/2026-Q1-governance-timeline-summary.md`
