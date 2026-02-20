"""
统一日志配置模块。
支持标准格式和 JSON 格式，文件日志和控制台日志分离。
"""
import json
import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON 格式的日志格式化器，带敏感信息过滤。"""

    # 需要过滤的敏感字段模式
    SENSITIVE_PATTERNS = [
        r'api[_-]?key', r'api[_-]?secret', r'password',
        r'token', r'private[_-]?key', r'secret', r'auth'
    ]

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """过滤敏感信息。"""
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(re.search(p, key_lower) for p in self.SENSITIVE_PATTERNS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize(value)
            else:
                sanitized[key] = value
        return sanitized

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加额外字段（过滤敏感信息）
        if hasattr(record, "extra_data"):
            log_data["extra"] = self._sanitize(record.extra_data)

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class StandardFormatter(logging.Formatter):
    """标准格式的日志格式化器。"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    force: bool = True,
) -> None:
    """
    配置全局日志系统。

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径
        log_format: 格式类型 (json 或 standard)
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        force: 是否强制清除现有处理器，默认为 True。设置为 False 可在测试时保留现有处理器（如 pytest 的 caplog）
    """
    # 从环境变量读取配置，默认日志路径使用项目根目录
    level = level or os.getenv("LOG_LEVEL", "INFO")
    default_log_file = str(Path(__file__).resolve().parent.parent / "logs" / "corp.log")
    log_file = log_file or os.getenv("LOG_FILE", default_log_file)
    log_format = log_format or os.getenv("LOG_FORMAT", "standard")

    # 解析日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 根日志配置
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 清除现有处理器 (unless force=False for testing)
    if force:
        root_logger.handlers = []

    # 创建格式器
    if log_format.lower() == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = StandardFormatter()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        # 确保日志目录存在并设置权限
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 设置目录权限为 750 (rwxr-x---)
        os.chmod(log_path.parent, 0o750)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # 设置日志文件权限
        if os.path.exists(log_file):
            os.chmod(log_file, 0o640)  # rw-r-----

    # 设置第三方库日志级别
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"Logging configured: level={level}, format={log_format}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器。"""
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> Dict[str, Any]:
    """
    创建额外的日志数据字典。

    Usage:
        logger.info("Message", extra=log_extra(instrument="BTC", price=50000))
    """
    return {"extra_data": kwargs}


# 便捷函数
def debug(msg: str, **kwargs: Any) -> None:
    """输出 DEBUG 级别日志。"""
    logger = logging.getLogger("corp")
    if kwargs:
        logger.debug(msg, extra=log_extra(**kwargs))
    else:
        logger.debug(msg)


def info(msg: str, **kwargs: Any) -> None:
    """输出 INFO 级别日志。"""
    logger = logging.getLogger("corp")
    if kwargs:
        logger.info(msg, extra=log_extra(**kwargs))
    else:
        logger.info(msg)


def warning(msg: str, **kwargs: Any) -> None:
    """输出 WARNING 级别日志。"""
    logger = logging.getLogger("corp")
    if kwargs:
        logger.warning(msg, extra=log_extra(**kwargs))
    else:
        logger.warning(msg)


def error(msg: str, **kwargs: Any) -> None:
    """输出 ERROR 级别日志。"""
    logger = logging.getLogger("corp")
    if kwargs:
        logger.error(msg, extra=log_extra(**kwargs))
    else:
        logger.error(msg)


def exception(msg: str, **kwargs: Any) -> None:
    """输出 EXCEPTION 级别日志（包含异常堆栈）。"""
    logger = logging.getLogger("corp")
    if kwargs:
        logger.exception(msg, extra=log_extra(**kwargs))
    else:
        logger.exception(msg)
