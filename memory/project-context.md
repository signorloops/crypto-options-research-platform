# CORP 项目上下文（长期）

> 最后更新：2026-03-01
> 用途：作为会话恢复的最小事实集，避免记录易过期的测试数量、覆盖率和临时修复清单。

## 项目定位

- 项目名：CORP（Crypto Options Research Platform）
- 仓库根目录：`<PROJECT_ROOT>/corp`
- 目标：围绕币本位期权研究，提供数据接入、定价与波动率建模、策略回测、治理与发布门禁。

## 当前文档入口

- 总览入口：`README.md`
- 导航入口：`docs/GUIDE.md`
- 计划入口：`docs/plans/README.md`
- 归档入口：`docs/archive/README.md`

## 文档治理约定

1. `docs/plans/` 仅保留正在执行的计划。
2. 完结/停用计划移动到 `docs/archive/plans/`。
3. `README.md` 与 `docs/GUIDE.md` 只保留必要入口，避免重复复制长说明。
4. 历史报告与旧手册统一放在 `docs/archive/`。

## 治理与验收命令（基线）

```bash
make complexity-audit
make complexity-audit-regression
make weekly-operating-audit
make weekly-close-gate
make live-deviation-snapshot
make docs-link-check
```

## 代码提交前最小检查

```bash
pytest -q -m "not integration"
make docs-link-check
make complexity-audit
```

## 变更边界

- 优先做“可验证、可回滚、低耦合”的精简（文档去重、入口收敛、无效资产清理）。
- 任何会影响运行逻辑的改动，必须配套测试或回归命令验证。
- 清理动作不得破坏 README/GUIDE/计划索引之间的链接闭环。

## 备注

- 本文件只记录长期稳定事实；阶段性进展写入 `docs/reports/` 或计划文档。
