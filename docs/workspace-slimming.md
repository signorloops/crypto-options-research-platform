# Workspace Slimming Playbook

## Why

本项目的主要体积来源通常不是源码，而是：

1. 本地虚拟环境（`venv/`, `.venv/`, `env/`）
2. 生成产物（`artifacts/`, `results/`）
3. 缓存目录（`.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `__pycache__/`）

## Dependency Profiles

默认安装采用精简开发栈：

```bash
pip install -e ".[dev]"
```

完整栈（包含 ML + Notebook + 加速器）：

```bash
pip install -e ".[dev,full]"
```

可选分组：

- `accelerated`: `numba`, `py_vollib`, `fastparquet`
- `ml`: `torch`, `xgboost`
- `notebook`: `jupyter`, `ipywidgets`

## Cleanup Commands

查看清理计划（默认 dry-run）：

```bash
make workspace-slim-report
```

清理安全目标（缓存 + artifacts + logs + untracked results）：

```bash
make workspace-slim-clean
```

连同本地虚拟环境一起清理：

```bash
make workspace-slim-clean-venv
```

## Notes

1. `workspace-slim-clean` 默认不会删除 `venv`。  
2. `workspace-slim-clean` 仅清理 `results/` 下 untracked 文件。  
3. 若你使用多个仓库副本，建议只保留一个激活开发副本，其余通过 git worktree 临时创建。  
