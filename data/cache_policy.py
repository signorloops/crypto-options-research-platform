"""Centralized cache TTL and invalidation policy."""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 1)


GREEKS_TTL_SECONDS = _env_int("CACHE_TTL_GREEKS_SECONDS", 30)
IV_TTL_SECONDS = _env_int("CACHE_TTL_IV_SECONDS", 30)
IV_TERM_TTL_SECONDS = _env_int("CACHE_TTL_IV_TERM_SECONDS", 300)
ORDERBOOK_TTL_SECONDS = _env_int("CACHE_TTL_ORDERBOOK_SECONDS", 1)
TICKER_TTL_SECONDS = _env_int("CACHE_TTL_TICKER_SECONDS", 5)

REALTIME_CACHE_PATTERNS = {
    "greeks": "greeks:{instrument}",
    "iv": "iv:{instrument}",
    "orderbook": "orderbook:{instrument}",
    "ticker": "ticker:{instrument}",
    "iv_term": "iv_term:{underlying}",
}


def invalidation_patterns(
    *,
    instrument: Optional[str] = None,
    underlying: Optional[str] = None,
) -> List[str]:
    """Build deterministic Redis invalidation patterns."""
    patterns: List[str] = []
    if instrument:
        patterns.extend(
            [
                REALTIME_CACHE_PATTERNS["greeks"].format(instrument=instrument),
                REALTIME_CACHE_PATTERNS["iv"].format(instrument=instrument),
                REALTIME_CACHE_PATTERNS["orderbook"].format(instrument=instrument),
                REALTIME_CACHE_PATTERNS["ticker"].format(instrument=instrument),
            ]
        )
    else:
        patterns.extend(["greeks:*", "iv:*", "orderbook:*", "ticker:*"])

    if underlying:
        patterns.append(REALTIME_CACHE_PATTERNS["iv_term"].format(underlying=underlying))
    else:
        patterns.append("iv_term:*")
    return patterns


def realtime_ttls() -> Dict[str, int]:
    """Expose TTLs for observability and docs/UI."""
    return {
        "greeks": GREEKS_TTL_SECONDS,
        "iv": IV_TTL_SECONDS,
        "iv_term": IV_TERM_TTL_SECONDS,
        "orderbook": ORDERBOOK_TTL_SECONDS,
        "ticker": TICKER_TTL_SECONDS,
    }
