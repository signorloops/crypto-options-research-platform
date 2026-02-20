"""
Local data cache for storing and retrieving market data.
Uses Parquet format for efficient storage and retrieval.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


import os


class DataCache:
    """
    File-based cache for market data.

    Organizes data by:
    - data/raw/{exchange}/{data_type}/{instrument}/{YYYY-MM-DD}.parquet
    - data/processed/{feature_set}/{instrument}/{YYYY-MM-DD}.parquet
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            # Default to current project root/data/cache or env var
            base_dir = os.getenv("CORP_CACHE_DIR")
            if base_dir is None:
                base_dir = Path(__file__).resolve().parent.parent / "data" / "cache"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def _get_cache_path(
        self,
        exchange: str,
        data_type: str,
        instrument: str,
        date: datetime,
        processed: bool = False
    ) -> Path:
        """Generate cache file path."""
        base = self.processed_dir if processed else self.raw_dir

        # Sanitize instrument name for filesystem
        safe_instrument = instrument.replace("/", "_").replace("-", "_")

        dir_path = base / exchange / data_type / safe_instrument
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{date.strftime('%Y-%m-%d')}.parquet"
        return dir_path / filename

    def get(
        self,
        exchange: str,
        data_type: str,
        instrument: str,
        start: datetime,
        end: datetime,
        processed: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data for date range.

        Returns:
            DataFrame if all dates cached, None otherwise
        """
        if start > end:
            raise ValueError("start must be <= end")

        frames = []

        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_date:
            cache_path = self._get_cache_path(
                exchange, data_type, instrument, current, processed
            )

            if not cache_path.exists():
                return None  # Missing data

            try:
                df = pd.read_parquet(cache_path)
                frames.append(df)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                # Handle corrupted or unreadable cache files
                logger.warning("Failed to read cache file", extra=log_extra(path=str(cache_path), error=str(e)))
                return None

            current += timedelta(days=1)

        if not frames:
            return None

        # Concatenate and filter to exact time range
        combined = pd.concat(frames, ignore_index=True)

        if 'timestamp' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
            # If start and end are the same day (midnight), treat end as end of day
            if start == end and start.hour == 0 and start.minute == 0:
                end = end + timedelta(days=1) - timedelta(microseconds=1)
            combined = combined[
                (combined['timestamp'] >= start) &
                (combined['timestamp'] <= end)
            ]

        return combined.reset_index(drop=True)

    def put(
        self,
        df: pd.DataFrame,
        exchange: str,
        data_type: str,
        instrument: str,
        date: datetime,
        processed: bool = False
    ) -> None:
        """Store data to cache."""
        # Check if data spans multiple dates
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            unique_dates = df_copy['timestamp'].dt.date.nunique()
            if unique_dates > 1:
                # Use put_range for multi-day data
                self.put_range(df_copy, exchange, data_type, instrument, processed)
                return

        cache_path = self._get_cache_path(
            exchange, data_type, instrument, date, processed
        )

        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df.to_parquet(cache_path, compression='zstd')

    def put_range(
        self,
        df: pd.DataFrame,
        exchange: str,
        data_type: str,
        instrument: str,
        processed: bool = False
    ) -> None:
        """Store data spanning multiple days, split by date."""
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        for date, group in df.groupby('date'):
            group = group.drop('date', axis=1)
            self.put(
                group,
                exchange,
                data_type,
                instrument,
                datetime.combine(date, datetime.min.time()),
                processed
            )

    def exists(
        self,
        exchange: str,
        data_type: str,
        instrument: str,
        start: datetime,
        end: datetime,
        processed: bool = False
    ) -> bool:
        """Check if data is fully cached."""
        if start > end:
            raise ValueError("start must be <= end")

        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_date:
            cache_path = self._get_cache_path(
                exchange, data_type, instrument, current, processed
            )
            if not cache_path.exists():
                return False
            current += timedelta(days=1)

        return True

    def list_available(
        self,
        exchange: str,
        data_type: str,
        instrument: str,
        processed: bool = False
    ) -> List[datetime]:
        """List available cached dates."""
        base = self.processed_dir if processed else self.raw_dir
        safe_instrument = instrument.replace("/", "_").replace("-", "_")
        dir_path = base / exchange / data_type / safe_instrument

        if not dir_path.exists():
            return []

        dates = []
        for f in dir_path.glob("*.parquet"):
            try:
                date = datetime.strptime(f.stem, "%Y-%m-%d")
                dates.append(date)
            except ValueError:
                continue

        return sorted(dates)

    def get_cache_info(self) -> Dict:
        """Get cache statistics."""
        info = {
            "raw_size_mb": 0,
            "processed_size_mb": 0,
            "total_files": 0,
            "instruments": {}
        }

        for base, label in [(self.raw_dir, "raw"), (self.processed_dir, "processed")]:
            if not base.exists():
                continue

            for path in base.rglob("*.parquet"):
                size_mb = path.stat().st_size / (1024 * 1024)
                info["total_files"] += 1

                if label == "raw":
                    info["raw_size_mb"] += size_mb
                else:
                    info["processed_size_mb"] += size_mb

        return info

    def clear_cache(self, exchange: Optional[str] = None, older_than_days: Optional[int] = None) -> None:
        """Clear cached data."""
        import shutil

        if exchange:
            # Clear specific exchange
            for base in [self.raw_dir, self.processed_dir]:
                exchange_dir = base / exchange
                if exchange_dir.exists():
                    if older_than_days:
                        # Selective deletion
                        cutoff = datetime.now() - timedelta(days=older_than_days)
                        for f in exchange_dir.rglob("*.parquet"):
                            try:
                                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
                                if file_date < cutoff:
                                    f.unlink()
                            except ValueError:
                                pass
                    else:
                        # Delete all
                        shutil.rmtree(exchange_dir)
        else:
            # Clear all
            if older_than_days:
                cutoff = datetime.now() - timedelta(days=older_than_days)
                for base in [self.raw_dir, self.processed_dir]:
                    for f in base.rglob("*.parquet"):
                        try:
                            file_date = datetime.strptime(f.stem, "%Y-%m-%d")
                            if file_date < cutoff:
                                f.unlink()
                        except ValueError:
                            pass
            else:
                shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(parents=True, exist_ok=True)


class DataManager:
    """
    High-level data manager combining download and cache.
    """

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()

    async def get_data(
        self,
        exchange: str,
        data_type: str,
        instrument: str,
        start: datetime,
        end: datetime,
        downloader = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get data with automatic caching.

        Args:
            exchange: Exchange name
            data_type: Type of data (trades, orderbook, etc.)
            instrument: Instrument name
            start: Start time
            end: End time
            downloader: Async function to download missing data
            use_cache: Whether to use cache

        Returns:
            DataFrame with requested data
        """
        if start > end:
            raise ValueError("start must be <= end")

        # Check cache. If read fails unexpectedly, fall back to downloader.
        if use_cache and self.cache.exists(exchange, data_type, instrument, start, end):
            cached = self.cache.get(exchange, data_type, instrument, start, end)
            if cached is not None:
                return cached
            logger.warning(
                "Cache exists but read returned no data; falling back to downloader",
                extra=log_extra(
                    exchange=exchange,
                    data_type=data_type,
                    instrument=instrument,
                    start=start.isoformat(),
                    end=end.isoformat(),
                ),
            )

        # Download if needed
        if downloader is None:
            raise ValueError("Data not cached and no downloader provided")

        data = await downloader(instrument, start, end)

        # Cache result
        if use_cache and data is not None and len(data) > 0:
            self.cache.put_range(data, exchange, data_type, instrument)

        return data
