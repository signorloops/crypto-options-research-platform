# 部署指南

## 生产环境检查清单

### 1. 安全
- [ ] API Keys 存储在环境变量，不在代码中
- [ ] `.env` 文件已添加到 `.gitignore`
- [ ] 日志文件权限设置为 640
- [ ] 日志敏感信息过滤已启用
- [ ] 运行 `grep -r "api_key\|secret" --include="*.py" .` 确认无硬编码

### 2. 性能
- [ ] 数据缓存目录有足够磁盘空间 (>100GB)
- [ ] 内存 >= 8GB（用于大数据集）
- [ ] 网络带宽稳定（WebSocket 需要低延迟）
- [ ] 使用 SSD 存储缓存数据

### 3. 监控
- [ ] 日志轮转配置（防止磁盘满）
- [ ] 异常告警机制（Slack/邮件）
- [ ] API 健康检查端点
- [ ] 关键指标监控（延迟、成功率）

## Docker 部署

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖
COPY pyproject.toml .
COPY core/ ./core/
COPY data/ ./data/
COPY strategies/ ./strategies/
COPY research/ ./research/
COPY utils/ ./utils/

# 安装 Python 依赖
RUN pip install --no-cache-dir -e "."

# 创建非 root 用户
RUN useradd -m -u 1000 corp && chown -R corp:corp /app
USER corp

# 挂载点
VOLUME ["/app/data/cache", "/app/logs"]

# 入口
CMD ["python", "-m", "execution.service_runner"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  corp:
    build: .
    container_name: corp-research
    environment:
      - LOG_LEVEL=INFO
      - LOG_FILE=/app/logs/corp.log
      - DERIBIT_API_KEY=${DERIBIT_API_KEY}
      - DERIBIT_API_SECRET=${DERIBIT_API_SECRET}
      - OKX_API_KEY=${OKX_API_KEY}
      - OKX_API_SECRET=${OKX_API_SECRET}
    volumes:
      - ./data/cache:/app/data/cache
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
    restart: unless-stopped
    networks:
      - corp-network

  # 可选：Redis 缓存
  redis:
    image: redis:7-alpine
    container_name: corp-redis
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - corp-network

  # 可选：监控
  prometheus:
    image: prom/prometheus:latest
    container_name: corp-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    networks:
      - corp-network

networks:
  corp-network:
    driver: bridge

volumes:
  redis-data:
```

## 统一 Service 入口（Runbook）

所有部署形态统一使用同一个 Python 入口：

```bash
python -m execution.service_runner
```

通过环境变量区分服务角色：

| 服务角色 | `SERVICE_NAME` | 端口变量 | 默认端口 |
|---|---|---|---|
| 交易引擎 | `trading-engine` | `TRADING_ENGINE_PORT` | `8080` |
| 风控服务 | `risk-monitor` | `RISK_MONITOR_PORT` | `8081` |
| 行情采集 | `market-data-collector` | `MARKET_DATA_COLLECTOR_PORT` | `8082` |

示例：

```bash
# 交易引擎
SERVICE_NAME=trading-engine TRADING_ENGINE_PORT=8080 python -m execution.service_runner

# 风控
SERVICE_NAME=risk-monitor RISK_MONITOR_PORT=8081 python -m execution.service_runner

# 行情采集
SERVICE_NAME=market-data-collector MARKET_DATA_COLLECTOR_PORT=8082 python -m execution.service_runner
```

说明：
- `execution.trading_engine` / `execution.risk_monitor` / `execution.market_data_collector` 仅保留兼容导入，不再作为部署入口。
- 提交前可执行 `make check-service-entrypoint`，防止部署脚本误用旧入口。

## 系统服务 (systemd)

### /etc/systemd/system/corp.service
```ini
[Unit]
Description=CORP Research Platform
After=network.target

[Service]
Type=simple
User=corp
Group=corp
WorkingDirectory=/opt/corp
Environment=PYTHONPATH=/opt/corp
EnvironmentFile=/opt/corp/.env
ExecStart=/opt/corp/venv/bin/python -m execution.service_runner
Restart=always
RestartSec=10

# 资源限制
LimitAS=4G
LimitRSS=4G
LimitNOFILE=65535

# 日志
StandardOutput=append:/var/log/corp/output.log
StandardError=append:/var/log/corp/error.log

[Install]
WantedBy=multi-user.target
```

### 启用服务
```bash
sudo systemctl daemon-reload
sudo systemctl enable corp
sudo systemctl start corp
sudo systemctl status corp
```

## 日志管理

### 日志轮转 (logrotate)
```bash
# /etc/logrotate.d/corp
/var/log/corp/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 640 corp corp
    sharedscripts
    postrotate
        systemctl reload corp
    endscript
}
```

## 备份策略

### 数据备份
```bash
#!/bin/bash
# backup.sh

# 配置（可通过环境变量覆盖）
BACKUP_ROOT="${BACKUP_ROOT:-/backup/corp}"
CORP_DATA_DIR="${CORP_DATA_DIR:-/opt/corp}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# 验证配置
if [[ -z "$BACKUP_ROOT" ]] || [[ "$BACKUP_ROOT" == "/" ]]; then
    echo "Error: BACKUP_ROOT not set or unsafe"
    exit 1
fi

# 创建备份目录
BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR" || { echo "Failed to create backup dir"; exit 1; }

# 备份缓存数据
if [[ -d "$CORP_DATA_DIR/data/cache" ]]; then
    tar czf "$BACKUP_DIR/cache.tar.gz" -C "$CORP_DATA_DIR" data/cache/
fi

# 备份配置
if [[ -f "$CORP_DATA_DIR/.env" ]]; then
    cp "$CORP_DATA_DIR/.env" "$BACKUP_DIR/"
fi
if [[ -f "$CORP_DATA_DIR/pyproject.toml" ]]; then
    cp "$CORP_DATA_DIR/pyproject.toml" "$BACKUP_DIR/"
fi

# 保留最近 N 天（安全的删除方式）
find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime +$RETENTION_DAYS -print0 | \
    while IFS= read -r -d '' dir; do
        echo "Removing old backup: $dir"
        rm -rf "$dir"
    done

echo "Backup completed: $BACKUP_DIR"
```

### 定时任务
```bash
# crontab -e
0 2 * * * /opt/corp/scripts/backup.sh
```

## 故障恢复

### 常见问题

#### 1. 内存不足
```bash
# 查看内存使用
free -h

# 增加交换空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 2. 磁盘满
```bash
# 检查磁盘使用
df -h

# 清理旧日志
find /var/log/corp -name "*.log" -mtime +7 -delete

# 清理旧缓存
find /opt/corp/data/cache -mtime +30 -delete
```

#### 3. API 限流
```bash
# 检查日志中的 429 错误
grep "429" /var/log/corp/error.log

# 临时降低请求频率
# 修改代码中的 rate limit 参数
```

### 紧急回滚
```bash
# 停止服务
sudo systemctl stop corp

# 恢复备份
tar xzf /backup/corp/20240101/cache.tar.gz -C /

# 重启服务
sudo systemctl start corp
```

## 性能调优

### 内核参数
```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
fs.file-max = 2097152
```

### Python 优化
```bash
# 使用 PyPy（如果兼容）
# 或启用 Python 优化
export PYTHONOPTIMIZE=1
```

## 监控指标

### 关键指标
| 指标 | 目标 | 告警阈值 |
|------|------|---------|
| API 延迟 | < 500ms | > 1s |
| WebSocket 重连次数 | 0 | > 5/小时 |
| 内存使用 | < 70% | > 85% |
| 磁盘使用 | < 80% | > 90% |
| 错误率 | < 1% | > 5% |

### 健康检查
```python
# health_check.py
import asyncio
import sys
from data.downloaders.deribit import DeribitClient

async def check():
    try:
        client = DeribitClient()
        async with client:
            tick = await client.get_tick("BTC-PERPETUAL")
            if tick.price > 0:
                print("HEALTHY")
                return 0
    except Exception as e:
        print(f"UNHEALTHY: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(check()))
```

## 扩展部署

### 多交易所部署
```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  corp-deribit:
    extends:
      file: docker-compose.yml
      service: corp
    environment:
      - EXCHANGE_FOCUS=deribit
    deploy:
      replicas: 2

  corp-okx:
    extends:
      file: docker-compose.yml
      service: corp
    environment:
      - EXCHANGE_FOCUS=okx
    deploy:
      replicas: 2
```

### Kubernetes (简化)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: corp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: corp
  template:
    metadata:
      labels:
        app: corp
    spec:
      containers:
      - name: corp
        image: corp:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        envFrom:
        - secretRef:
            name: corp-secrets
        volumeMounts:
        - name: cache
          mountPath: /app/data/cache
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: corp-cache
      - name: logs
        emptyDir: {}
```
