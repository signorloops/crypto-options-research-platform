"""
统一异常处理模块。

为整个项目提供一致的异常层次结构。
"""
from typing import Any, Optional


class CORPError(Exception):
    """项目基础异常类。"""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.code: Optional[str] = code

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class ValidationError(CORPError):
    """数据验证错误。"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        code: str = "VALIDATION_ERROR"
    ) -> None:
        self.field: Optional[str] = field
        self.value: Any = value
        if field is not None and value is not None:
            super().__init__(f"{field}={value}: {message}", code)
        else:
            super().__init__(message, code)


class PricingError(CORPError):
    """定价计算错误。"""

    def __init__(self, message: str, code: str = "PRICING_ERROR") -> None:
        super().__init__(message, code)


class APIError(CORPError):
    """API 调用错误。"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        code: str = "API_ERROR"
    ) -> None:
        self.status_code: Optional[int] = status_code
        super().__init__(message, code)


class DataError(CORPError):
    """数据获取或处理错误。"""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        code: str = "DATA_ERROR"
    ) -> None:
        self.source: Optional[str] = source
        super().__init__(message, code)


class ConfigurationError(CORPError):
    """配置错误。"""

    def __init__(self, message: str, code: str = "CONFIG_ERROR") -> None:
        super().__init__(message, code)


class StrategyError(CORPError):
    """策略执行错误。"""

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        code: str = "STRATEGY_ERROR"
    ) -> None:
        self.strategy_name: Optional[str] = strategy_name
        super().__init__(message, code)
