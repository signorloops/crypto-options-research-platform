"""Tests for cross-venue quote integration helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import quote_integration as qi


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_dataset_aligns_cross_minute_quotes_with_tolerance(tmp_path):
    cex_path = tmp_path / "cex.csv"
    defi_path = tmp_path / "defi.csv"
    _write_csv(
        cex_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:59Z,BTC-OPT,call,0.05,0.25,1200,okx\n"
        ),
    )
    _write_csv(
        defi_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:01:01Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
    )

    out = qi.build_cex_defi_deviation_dataset(
        cex_path,
        defi_path,
        align_tolerance_seconds=3,
    )
    assert len(out) == 1
    assert out.iloc[0]["market_price"] == pytest.approx(1200.0)
    assert out.iloc[0]["model_price"] == pytest.approx(1140.0)


def test_build_dataset_respects_tolerance_limit(tmp_path):
    cex_path = tmp_path / "cex.csv"
    defi_path = tmp_path / "defi.csv"
    _write_csv(
        cex_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:59Z,BTC-OPT,call,0.05,0.25,1200,okx\n"
        ),
    )
    _write_csv(
        defi_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:01:01Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
    )

    with pytest.raises(ValueError, match="No aligned CEX/DeFi rows"):
        qi.build_cex_defi_deviation_dataset(
            cex_path,
            defi_path,
            align_tolerance_seconds=1,
        )


def test_normalize_okx_option_summary_infers_option_type_and_defaults():
    now_ms = int(pd.Timestamp("2026-02-25T00:00:00Z").timestamp() * 1000)
    rows = [
        {
            "instId": "BTC-USD-260329-50000-C",
            "markPx": "1234.5",
            "expTime": str(now_ms + 30 * 24 * 3600 * 1000),
            "ts": str(now_ms),
        }
    ]

    normalized = qi._normalize_okx_option_summary(rows, underlying="BTC-USD")
    assert len(normalized) == 1
    assert normalized.iloc[0]["option_type"] == "call"
    assert normalized.iloc[0]["delta"] == pytest.approx(0.5)
    assert normalized.iloc[0]["venue"] == "okx"


@pytest.mark.asyncio
async def test_live_builder_uses_provider_and_returns_aligned_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    defi_path = tmp_path / "defi.csv"
    _write_csv(
        defi_path,
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:00:00Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
    )

    cex_normalized = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "price": 1200.0,
                "symbol": "BTC-OPT",
                "option_type": "call",
                "expiry_years": 0.05,
                "delta": 0.25,
                "venue": "okx",
                "expiry_bucket": 0.05,
                "delta_bucket": 0.25,
                "ts_bucket": pd.Timestamp("2024-01-01T00:00:00Z"),
            }
        ]
    )

    class FakeOKXClient:
        disconnected = False
        received_underlying = ""

        async def get_option_market_data(self, underlying: str = "BTC-USD"):
            FakeOKXClient.received_underlying = underlying
            return [{"mock": True}]

        async def disconnect(self) -> None:
            FakeOKXClient.disconnected = True

    monkeypatch.setattr("data.downloaders.okx.OKXClient", FakeOKXClient)
    monkeypatch.setattr(
        qi, "_normalize_okx_option_summary", lambda rows, underlying: cex_normalized
    )

    out = await qi.build_cex_defi_deviation_dataset_live(
        "okx",
        defi_path,
        underlying="BTC-USD",
    )
    assert len(out) == 1
    assert FakeOKXClient.received_underlying == "BTC-USD"
    assert FakeOKXClient.disconnected is True
