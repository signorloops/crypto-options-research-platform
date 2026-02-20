"""
Data module for market data acquisition and streaming (coin-margined options only).
"""
from data.cache import DataCache, DataManager
from data.downloaders.deribit import DeribitClient, DeribitDataDownloader
from data.downloaders.okx import OKXClient
from data.streaming import (
    DeribitStream,
    MultiExchangeStream,
    OKXStream,
    StreamConfig,
)

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
