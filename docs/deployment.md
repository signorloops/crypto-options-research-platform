# 部署指南（精简版）

## 1. 统一入口

所有服务统一通过：

```bash
python -m execution.service_runner
```

通过 `SERVICE_NAME` 区分角色：

| 服务 | `SERVICE_NAME` | 端口变量 | 默认端口 |
|---|---|---|---|
| 交易引擎 | `trading-engine` | `TRADING_ENGINE_PORT` | `8080` |
| 风控服务 | `risk-monitor` | `RISK_MONITOR_PORT` | `8081` |
| 行情采集 | `market-data-collector` | `MARKET_DATA_COLLECTOR_PORT` | `8082` |

示例：

```bash
SERVICE_NAME=trading-engine TRADING_ENGINE_PORT=8080 python -m execution.service_runner
SERVICE_NAME=risk-monitor RISK_MONITOR_PORT=8081 python -m execution.service_runner
SERVICE_NAME=market-data-collector MARKET_DATA_COLLECTOR_PORT=8082 python -m execution.service_runner
```

提交前校验旧入口误用：

```bash
make check-service-entrypoint
```

## 2. Docker 快速路径

```bash
docker compose up -d --build
```

建议挂载：

- `data/cache/`
- `logs/`

## 3. 生产前检查

1. 密钥仅在环境变量，不落库不入仓。
2. 日志轮转与敏感字段过滤已启用。
3. 资源满足最低要求（CPU/内存/磁盘）。
4. 健康检查可访问，关键告警链路已联通。

## 4. 运维常用命令

```bash
# 服务状态
systemctl status corp

# 启停
systemctl restart corp
systemctl stop corp

# 健康检查
curl -fsS http://127.0.0.1:8080/health
```

## 5. 回滚原则

1. 先停服务再恢复配置/数据快照。
2. 回滚后优先验证健康接口和最小回归。
3. 回滚动作必须记录在周治理产物中。

治理与审计流程见：

- `docs/governance-operations.md`
