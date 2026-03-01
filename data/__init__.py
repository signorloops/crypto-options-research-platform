"""Data module for market data acquisition and streaming.

This package intentionally exposes symbols lazily so importing ``data`` does
not require optional runtime dependencies (for example WebSocket clients).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Cache
    "DataCache",
    "DataManager",
    # Downloaders
    "DeribitClient",
    "DeribitDataDownloader",
    "OKXClient",
    # Streaming
    "DeribitStream",
    "OKXStream",
    "MultiExchangeStream",
    "StreamConfig",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DataCache": ("data.cache", "DataCache"),
    "DataManager": ("data.cache", "DataManager"),
    "DeribitClient": ("data.downloaders.deribit", "DeribitClient"),
    "DeribitDataDownloader": ("data.downloaders.deribit", "DeribitDataDownloader"),
    "OKXClient": ("data.downloaders.okx", "OKXClient"),
    "DeribitStream": ("data.streaming", "DeribitStream"),
    "OKXStream": ("data.streaming", "OKXStream"),
    "MultiExchangeStream": ("data.streaming", "MultiExchangeStream"),
    "StreamConfig": ("data.streaming", "StreamConfig"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'data' has no attribute '{name}'")

    module_name, symbol_name = target
    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
