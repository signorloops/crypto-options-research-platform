"""Additional unit tests for data validation and cleaning utilities."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.validation import DataCleaner, DataQualityReport, DataValidator


def test_validate_trades_detects_multiple_issues():
    validator = DataValidator()
    now = datetime(2026, 1, 1)
    df = pd.DataFrame(
        {
            "timestamp": [
                now,
                now + timedelta(minutes=1),
                now + timedelta(minutes=2),
                now + timedelta(minutes=3),
            ],
            "price": [100.0, -1.0, 220.0, 110.0],
            "size": [1.0, 1.0, 1.0, 1.0],
            "side": ["buy", "sell", "hold", "sell"],
            "trade_id": ["a", "b", "c", "c"],
        }
    )

    result = validator.validate_trades(df, max_price_gap=0.2)

    assert result.is_valid is False
    assert result.cleaned_data is not None
    assert any("duplicate trade IDs" in issue for issue in result.issues)
    assert any("negative/zero price" in issue for issue in result.issues)
    assert any("invalid side" in issue for issue in result.issues)


def test_validate_ohlcv_cleans_duplicates_nulls_and_bad_rows():
    validator = DataValidator()
    ts = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[1], ts[1], ts[2], ts[3]],
            "open": [100, 101, 101, 50, 100],
            "high": [101, 102, 102, 49, 101],  # row 4 invalid high < open
            "low": [99, 100, 100, 51, 99],  # row 4 invalid low > high
            "close": [100.5, 101.2, 101.2, 50.5, None],
            "volume": [10, 12, 12, 8, 9],
        }
    )

    result = validator.validate_ohlcv(df)

    assert result.is_valid is False
    assert result.cleaned_data is not None
    assert result.stats["gaps_found"] >= 0
    assert any("duplicate timestamps" in issue for issue in result.issues)
    assert any("Null values found" in issue for issue in result.issues)
    assert any("invalid OHLC logic" in issue for issue in result.issues)


def test_data_cleaner_resample_and_fill_and_align_paths():
    base = pd.date_range("2026-01-01", periods=6, freq="30min")
    df = pd.DataFrame(
        {
            "timestamp": base,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [0, 1, 2, 3, 4, 5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "volume": [10, 10, 10, 10, 10, 10],
        }
    )

    resampled = DataCleaner.resample_ohlcv(df, target_interval="1h")
    assert len(resampled) == 3

    gappy = df.drop(index=[2]).reset_index(drop=True)
    filled_ffill = DataCleaner.fill_gaps(gappy, method="forward_fill")
    filled_interp = DataCleaner.fill_gaps(gappy, method="interpolate")
    removed = DataCleaner.fill_gaps(gappy, method="remove")
    assert len(filled_ffill) >= len(gappy)
    assert len(filled_interp) >= len(gappy)
    assert len(removed) <= len(filled_ffill)

    left = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"), "x": [1, 2, 3, 4, 5]}
    )
    right = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"), "y": [10, 11, 12, 13, 14]}
    )
    aligned_left, aligned_right = DataCleaner.align_timestamps(left, right, method="ffill")
    assert len(aligned_left) == len(aligned_right)
    assert "x" in aligned_left.columns and "y" in aligned_right.columns


def test_data_cleaner_remove_outliers_and_errors():
    df = pd.DataFrame({"value": [1, 1, 1, 1, 100]})

    iqr = DataCleaner.remove_outliers(df, "value", method="iqr", threshold=1.5)
    zscore = DataCleaner.remove_outliers(df, "value", method="zscore", threshold=1.0)
    mad = DataCleaner.remove_outliers(df, "value", method="mad", threshold=3.0)

    assert len(iqr) < len(df)
    assert len(zscore) < len(df)
    assert len(mad) < len(df)

    with pytest.raises(ValueError):
        DataCleaner.remove_outliers(df, "value", method="unknown")


def test_data_quality_report_recommendations_and_unknown_type():
    reporter = DataQualityReport()
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1h"),
            "price": [100, 210, 220],
            "size": [1, 1, 1],
            "side": ["buy", "sell", "bad-side"],
            "trade_id": ["a", "a", "b"],
        }
    )

    report = reporter.generate_report(df, data_type="trades")
    assert report["is_valid"] is False
    assert report["issues_found"] >= 1
    assert len(report["recommendations"]) >= 1

    with pytest.raises(ValueError):
        reporter.generate_report(df, data_type="unknown")
