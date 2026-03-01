# 周执行清单模板（长期跟进项目）

> 用法：每周复制一份到周报或任务系统，按日填充并在周五归档。  
> 建议周期：每周一到周五固定执行。

自动化入口（生成 KPI 快照 + 风险例外 + 未完成任务清单）：

```bash
cd corp
make weekly-operating-audit
```

发布前一键硬门禁（要求 `READY_FOR_CLOSE`）：

```bash
make weekly-close-gate
```

生产偏离快照入口（可接入定时任务）：

```bash
cd corp
make live-deviation-snapshot
```

说明：该命令默认包含最小回归集（inverse/volatility/hawkes/dashboard）。
并自动生成近 7 天变更记录与回滚基线 tag（如存在）。
并自动串联生成 canary、收益归因、决策日志、人工签字包与线上/线下一致性回放报告。
其中签字包会读取一致性回放结果：`FAIL` 自动阻断，`PENDING_DATA` 或缺失状态进入待办项。
若仓库是 shallow clone（如 `actions/checkout` 默认深度），会标记 `变更记录完整` 为未完成；
请使用完整历史和 tag（例如 `fetch-depth: 0`，并确保 tag 可见）。

输出文件：

- `artifacts/weekly-operating-audit.md`
- `artifacts/weekly-operating-audit.json`
- `artifacts/weekly-pnl-attribution.md`
- `artifacts/weekly-pnl-attribution.json`
- `artifacts/weekly-canary-checklist.md`
- `artifacts/weekly-canary-checklist.json`
- `artifacts/weekly-adr-draft.md`
- `artifacts/weekly-decision-log.md`
- `artifacts/weekly-decision-log.json`
- `artifacts/weekly-signoff-pack.md`
- `artifacts/weekly-signoff-pack.json`
- `artifacts/weekly-manual-status.json`（首次运行自动生成模板）
- `artifacts/online-offline-consistency-replay.md`
- `artifacts/online-offline-consistency-replay.json`
- `artifacts/weekly-close-gate.md`（执行 `make weekly-close-gate` 时生成）
- `artifacts/weekly-close-gate.json`（执行 `make weekly-close-gate` 时生成）
  - 包含可直接复制到 PR 描述的 `PR Brief (Copy/Paste)` 段落与 `pr_brief` 字段
- `artifacts/live-deviation-snapshot.md`
- `artifacts/live-deviation-snapshot.json`
- `docs/templates/weekly-replay-template.md`

---

## 1. 周信息

- 周次：`YYYY-WW`  
- 时间范围：`YYYY-MM-DD` 到 `YYYY-MM-DD`  
- 本周 Owner：  
- 本周最高优先级目标（最多 3 项）：  
1.  
2.  
3.  

---

## 2. 周一（目标与门槛）

1. 更新上周 KPI 快照（PnL / Sharpe / MaxDD / VaR breach / fill calibration error）。  
2. 确认本周实验列表（最多 3 项，全部带实验编号）。  
3. 设定本周风险门槛（触发降级/回滚条件）。  

完成标记：
- [ ] KPI 快照更新  
- [ ] 实验编号分配完成  
- [ ] 风险门槛已确认  

---

## 3. 周二（实现与变更）

1. 代码实现或参数调优。  
2. 每个变更绑定：实验编号、数据窗口、配置版本、预期影响。  
3. 提交前通过最小回归测试。  

完成标记：
- [ ] 变更记录完整  
- [ ] 最小回归通过  
- [ ] 回滚版本已标记  

---

## 4. 周三（验证与审计）

1. 运行回测回归与线上/离线一致性检查。  
2. 运行日回归门禁（`make daily-regression`）。  
3. 生成风险例外（VaR/ES）报告。  
4. 运行复杂度治理基线（`make complexity-audit`）并确认 CI changed-files lint/type gate 通过。  
5. 对异常项进行归因与修复计划。  

完成标记：
- [ ] 一致性检查完成  
- [ ] 风险例外报告输出  
- [ ] 复杂度治理基线通过  
- [ ] 异常项已归因  

---

## 5. 周四（灰度与观察）

1. 小流量、小仓位灰度发布。  
2. 观察 24h：收益、风险、漂移告警、回滚信号。  
3. 对照 `artifacts/weekly-canary-checklist.md` 执行并签字。  
4. 达不到门槛立即降级或回滚。  

完成标记：
- [ ] 灰度发布完成  
- [ ] 24h 观察完成  
- [ ] 是否触发回滚已决策  

---

## 6. 周五（复盘与沉淀）

1. 固化 5 个交付物。  
2. 输出 ADR（决策、依据、风险、是否回滚）。  
3. 生成下周计划与阻塞项。  

完成标记：
- [ ] KPI 快照  
- [ ] 风险例外报告  
- [ ] 收益归因表（自动生成后确认）  
- [ ] 变更与回滚记录  
- [ ] ADR  

---

## 7. 关键模板

### A. 变更记录

- 实验编号：  
- 变更内容：  
- 数据窗口：  
- 配置版本：  
- 预期影响：  
- 回滚版本：  

### B. 风险例外记录

- 指标：  
- 阈值：  
- 实际值：  
- 影响范围：  
- 处置动作：  

### C. 收益归因记录

- spread capture：  
- adverse selection：  
- inventory cost：  
- hedging cost：  
- 结论：  
