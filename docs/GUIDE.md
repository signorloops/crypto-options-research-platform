# CORP 项目指南（精简导航版）

本文件是项目文档导航中心。
原则：先定位目标，再跳转到对应文档；避免在多个文档重复维护同一段说明。

## 1. 5 分钟起步

```bash
cd /path/to/crypto-options-research-platform
source venv/bin/activate
pip install -e ".[dev]"
pytest -q -m "not integration"
```

完成后按目标阅读：

- 快速上手：[`quickstart.md`](quickstart.md)
- 全局认知：[`../README.md`](../README.md)
- 系统结构：[`architecture.md`](architecture.md)

## 2. 按任务找文档

| 任务 | 入口文档 |
|---|---|
| 项目总览与能力边界 | [`../README.md`](../README.md) |
| 环境安装与第一个运行 | [`quickstart.md`](quickstart.md) |
| 架构与模块关系 | [`architecture.md`](architecture.md) |
| 理论与模型推导 | [`theory.md`](theory.md) |
| API/模块速查 | [`api.md`](api.md) |
| 运行示例 | [`examples.md`](examples.md) |
| Hawkes 对比实验 | [`hawkes_comparison_experiment.md`](hawkes_comparison_experiment.md) |
| 缓存与性能策略 | [`cache_strategy.md`](cache_strategy.md) |
| 研究看板说明 | [`dashboard.md`](dashboard.md) |
| 部署与运维 | [`deployment.md`](deployment.md) |
| 研究审计守门 | [`research-audit.md`](research-audit.md) |
| 复杂度治理规则 | [`complexity-governance.md`](complexity-governance.md) |
| 工作区瘦身流程 | [`workspace-slimming.md`](workspace-slimming.md) |
| 项目全景图 | [`project-map-mermaid.md`](project-map-mermaid.md) |
| 算法学习（入门） | [`算法与模型入门学习版.md`](算法与模型入门学习版.md) |
| 算法学习（深度） | [`算法与模型深度讲解.md`](算法与模型深度讲解.md) |
| 计划与执行状态 | [`plans/README.md`](plans/README.md) |

## 3. 按角色推荐路径

### 量化研究

1. [`../README.md`](../README.md)
2. [`quickstart.md`](quickstart.md)
3. [`hawkes_comparison_experiment.md`](hawkes_comparison_experiment.md)
4. [`theory.md`](theory.md)

### 工程开发

1. [`../README.md`](../README.md)
2. [`architecture.md`](architecture.md)
3. [`api.md`](api.md)
4. [`examples.md`](examples.md)

### 运维发布

1. [`deployment.md`](deployment.md)
2. [`dashboard.md`](dashboard.md)
3. [`plans/weekly-operating-checklist.md`](plans/weekly-operating-checklist.md)

## 4. 计划文档入口

- 计划索引：[`plans/README.md`](plans/README.md)
- 当前执行计划：
  - [`plans/2026-Q2-long-term-execution-roadmap.md`](plans/2026-Q2-long-term-execution-roadmap.md)
  - [`plans/weekly-operating-checklist.md`](plans/weekly-operating-checklist.md)
  - [`plans/2026-02-25-inverse-options-arxiv-implementation-plan.md`](plans/2026-02-25-inverse-options-arxiv-implementation-plan.md)
- 历史计划：[`archive/plans/README.md`](archive/plans/README.md)

## 5. 常用命令

```bash
# 测试
pytest -q -m "not integration"

# 文档与治理
make docs-link-check
make complexity-audit
make weekly-operating-audit
make weekly-close-gate

# 偏离快照
make live-deviation-snapshot
```

## 6. 文档维护规则

1. 本文件只做导航，不复制大段教程。
2. 深入内容统一维护在专题文档，避免多处更新。
3. `docs/plans/` 只保留在执行计划，历史内容移入 `docs/archive/plans/`。
4. 修改导航后，执行 `make docs-link-check` 验证链接。
