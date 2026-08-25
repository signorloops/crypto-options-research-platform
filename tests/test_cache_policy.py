"""Tests for cache policy TTL and invalidation helpers."""

import importlib


def test_invalidation_patterns_with_and_without_scope():
    from data.cache_policy import invalidation_patterns

    scoped = invalidation_patterns(instrument="BTC-TEST", underlying="BTC")
    assert scoped == [
        "greeks:BTC-TEST",
        "iv:BTC-TEST",
        "orderbook:BTC-TEST",
        "ticker:BTC-TEST",
        "iv_term:BTC",
    ]

    global_patterns = invalidation_patterns()
    assert global_patterns == ["greeks:*", "iv:*", "orderbook:*", "ticker:*", "iv_term:*"]


def test_invalidation_patterns_instrument_only_keeps_other_underlyings():
    """A single-instrument invalidation must not wipe every underlying's term structure.

    Previously `iv_term:*` was appended whenever `underlying` was omitted, so
    invalidating one option cleared the IV term structures of all underlyings.
    Only a full wipe (no instrument and no underlying) may match `iv_term:*`.
    """
    from data.cache_policy import invalidation_patterns

    instrument_only = invalidation_patterns(instrument="BTC-TEST")
    assert instrument_only == [
        "greeks:BTC-TEST",
        "iv:BTC-TEST",
        "orderbook:BTC-TEST",
        "ticker:BTC-TEST",
    ]
    assert "iv_term:*" not in instrument_only

    # Underlying-only scoping still invalidates just that underlying.
    underlying_only = invalidation_patterns(underlying="ETH")
    assert underlying_only == [
        "greeks:*",
        "iv:*",
        "orderbook:*",
        "ticker:*",
        "iv_term:ETH",
    ]


def test_realtime_ttls_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("CACHE_TTL_GREEKS_SECONDS", "44")
    monkeypatch.setenv("CACHE_TTL_IV_SECONDS", "55")
    monkeypatch.setenv("CACHE_TTL_IV_TERM_SECONDS", "66")
    monkeypatch.setenv("CACHE_TTL_ORDERBOOK_SECONDS", "2")
    monkeypatch.setenv("CACHE_TTL_TICKER_SECONDS", "7")

    import data.cache_policy as cache_policy

    importlib.reload(cache_policy)
    ttls = cache_policy.realtime_ttls()
    assert ttls == {
        "greeks": 44,
        "iv": 55,
        "iv_term": 66,
        "orderbook": 2,
        "ticker": 7,
    }
