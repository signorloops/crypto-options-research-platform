"""Configuration settings using Pydantic."""
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: Optional[str], default_relative: str) -> str:
    """Resolve path to absolute, using env var or project root as base."""
    if path:
        return str(Path(path).expanduser().resolve())
    return str(PROJECT_ROOT / default_relative)


class Settings(BaseSettings):
    """CORP application settings."""

    # Environment
    environment: Literal["development", "testing", "staging", "production"] = "development"

    # Logging - supports LOG_FILE env var, defaults to project_root/logs/corp.log
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "json"

    # Data - supports CORP_CACHE_DIR env var
    data_cache_dir: str = os.getenv("CORP_CACHE_DIR", str(PROJECT_ROOT / "data" / "cache"))
    redis_url: Optional[str] = None

    # Trading
    default_exchange: str = "deribit"
    paper_trading: bool = True

    # Risk Management
    max_position_size: float = 1000.0
    max_drawdown_pct: float = 0.1

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()

    @field_validator('log_file')
    @classmethod
    def set_default_log_file(cls, v: Optional[str]) -> str:
        """Set default log file path if not provided."""
        if v is None:
            return str(PROJECT_ROOT / "logs" / "corp.log")
        return str(Path(v).expanduser().resolve())

    @field_validator('data_cache_dir')
    @classmethod
    def set_default_cache_dir(cls, v: str) -> str:
        """Ensure cache directory is absolute path."""
        if not Path(v).is_absolute():
            return str(PROJECT_ROOT / v)
        return v

    @field_validator('max_drawdown_pct')
    @classmethod
    def validate_max_drawdown(cls, v: float) -> float:
        """Validate max drawdown is within reasonable range."""
        if not 0 < v <= 1:
            raise ValueError("max_drawdown_pct must be between 0 and 1 (0% to 100%)")
        return v

    @field_validator('max_position_size')
    @classmethod
    def validate_max_position(cls, v: float) -> float:
        """Validate max position size is positive."""
        if v <= 0:
            raise ValueError("max_position_size must be positive")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
