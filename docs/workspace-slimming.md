# Workspace Slimming Playbook

## Why

本项目的主要体积来源通常不是源码，而是：

1. 本地虚拟环境（`venv/`, `.venv/`, `env/`）
2. 生成产物（`artifacts/`, `results/`）
3. 缓存目录（`.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `__pycache__/`）
4. 本地元数据文件（`.coverage*`, `.DS_Store`）

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

说明：上述 Makefile 目标默认开启多工作树扫描（`--all-worktrees`），会同时覆盖同一仓库下的所有 `git worktree`。

清理安全目标（缓存 + artifacts + logs + `.coverage*` + `.DS_Store` + untracked results）：

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
3. `.DS_Store` 清理会跳过 `.git/` 目录。  
3. 如需只扫描当前目录，可直接运行脚本并显式关闭多工作树行为（不传 `--all-worktrees`）。  
