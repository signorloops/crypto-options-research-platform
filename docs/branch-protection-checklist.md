# Branch Protection Checklist (master)

更新时间：2026-02-24（基于最近一次成功运行的 check 名称）。

## 1. Required Status Checks

建议在 `master` 分支保护规则里勾选以下必过检查：

1. `complexity-audit`
2. `security`
3. `test (3.9)`
4. `test (3.10)`
5. `test (3.11)`
6. `docker`

说明：
- `complexity-audit` 来自 `Complexity Governance` workflow 的 job 名称。
- 其余检查来自 `CI` workflow 的 job 名称。
- `Research Audit` 是周度/手动研究守门，默认不建议设为 merge 必过（避免阻塞日常开发）。

## 2. GitHub UI 配置步骤

1. 进入仓库 `Settings` -> `Branches`。
2. 在 `Branch protection rules` 新建或编辑 `master` 规则。
3. 启用：
- `Require a pull request before merging`
- `Require status checks to pass before merging`
4. 在 status checks 列表中添加上面的 6 个检查名。
5. 视需要启用：
- `Require branches to be up to date before merging`
- `Require conversation resolution before merging`

## 3. 快速核验命令

```bash
# 查看最近 CI job 名称（用于核对 required checks 文案）
gh run list --workflow CI --limit 1
gh run view <run-id> --json jobs

# 查看 complexity check 名称
gh run list --workflow "Complexity Governance" --limit 1
gh run view <run-id> --json jobs
```

## 4. 变更后回归检查

1. 推送一个文档小改动分支，创建 PR。
2. 确认以上 required checks 都出现且能完成。
3. 确认 PR 页面不再出现 `Expected — Waiting for status to be reported` 的永久等待状态。

