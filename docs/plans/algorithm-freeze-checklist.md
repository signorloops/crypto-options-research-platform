# Algorithm Freeze Checklist

目标：当核心算法链路达到可发布状态时，给出统一的“停止继续优化”判定标准，避免无限迭代。

## 1. Freeze Exit Criteria

以下条件全部满足时，视为达到冻结标准：

- `pytest -q -m "not integration"` 通过
- 文档链接检查通过（`make docs-link-check`）
- 分支命名守卫通过（`make branch-name-guard`）
- 复杂度回归守卫通过（`make complexity-audit-regression`）
- 算法性能基线通过（`make algorithm-performance-baseline`）
- 日常回归门通过（`make daily-regression`）

## 2. One-Command Freeze Verification

```bash
make algorithm-freeze-check
```

该命令会按顺序执行上述冻结门禁，任何一步失败即返回非零退出码。

## 3. Post-Freeze Allowed Changes

冻结后仅建议接受以下变更：

- 明确的缺陷修复（bugfix）
- 风险控制/监控守卫增强
- 不改变行为的重构（并通过全量回归）

## 4. Post-Freeze Forbidden Changes

冻结后应避免以下变更，除非重新开启优化周期：

- 调参与模型结构升级
- 指标口径变更
- 回测/治理阈值口径变更

## 5. Reopen Rule

出现以下任一情况时，可重新开启优化周期：

- 线上偏离告警持续超阈值
- 新市场结构变化导致既有假设失效
- 冻结后回归门出现系统性失败
