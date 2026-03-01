# Usage Examples Summary

本文件整合常用代码示例与 Hawkes 对比实验入口，替代历史长文档。

## 1. 数据与缓存

推荐入口：`IntegratedDataManager`  
核心模块：`data/cache.py`, `data/duckdb_cache.py`, `data/redis_cache.py`

## 2. 定价与波动率

常见路径：

- `research/pricing/inverse_options.py`
- `research/volatility/historical.py`
- `research/volatility/implied.py`

## 3. 回测与策略对比

Hawkes 对比框架：

- `research/backtest/hawkes_comparison.py`
- `tests/test_hawkes_comparison.py`
- `notebooks/06_hawkes_backtest_comparison.ipynb`

## 4. 看板与产物

```bash
python -m execution.research_dashboard
make live-deviation-snapshot
```

## 5. 建议阅读顺序

1. `quickstart.md`
2. `architecture.md`
3. `api.md`
4. 本文档
