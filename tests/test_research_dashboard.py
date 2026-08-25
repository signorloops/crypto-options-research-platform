"""Tests for the research dashboard web app."""

import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from execution.research_dashboard import create_dashboard_app


def test_dashboard_renders_parquet_with_named_index(tmp_path):
    # Regression: reset_index().rename(columns={"index": "index"}) only works
    # for an unnamed RangeIndex; a parquet round-trip of set_index("timestamp")
    # yields a named index and previously crashed px.line with
    # "Value of 'x' is not the name of a column".
    df = pd.DataFrame(
        {"equity": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h").rename("timestamp"),
    )
    parquet_path = tmp_path / "equity_curve.parquet"
    df.to_parquet(parquet_path)

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        files_response = client.get("/api/files")
        html_response = client.get("/")

    assert files_response.status_code == 200
    assert html_response.status_code == 200
    assert "CORP Research Dashboard" in html_response.text
    assert "Return Distribution" in html_response.text


def test_dashboard_renders_csv_with_index_column_already_present(tmp_path):
    # Regression: when the frame already has an "index" data column the
    # reset-index column cannot also be named "index"; the overview chart
    # must still render instead of colliding.
    csv_path = tmp_path / "backtest.csv"
    csv_path.write_text(
        "index,equity\n" "5,100\n" "6,101\n" "7,102\n",
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "CORP Research Dashboard" in response.text


def test_available_result_files_skips_vanished_files(tmp_path, monkeypatch):
    # Regression: stat() raised FileNotFoundError (-> HTTP 500) when a result
    # file was deleted between glob() and the mtime sort.
    from pathlib import Path

    from execution.dashboard.data_helpers import available_result_files

    kept = tmp_path / "kept.csv"
    kept.write_text("equity\n100\n", encoding="utf-8")
    (tmp_path / "vanished.csv").write_text("equity\n100\n", encoding="utf-8")

    original_stat = Path.stat

    def failing_stat(self, *args, **kwargs):
        if self.name == "vanished.csv":
            raise FileNotFoundError(str(self))
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    files = available_result_files(tmp_path)
    monkeypatch.setattr(Path, "stat", original_stat)

    assert [path.name for path in files] == ["kept.csv"]


def test_dashboard_deviation_api_buckets_unknown_expiry_separately(tmp_path):
    # Regression: NaN expiry previously defaulted to 0 days and polluted the
    # "<=7D" bucket; it now lands in a distinct "UNKNOWN" bucket.
    csv_path = tmp_path / "options_deviation.csv"
    csv_path.write_text(
        (
            "timestamp,exchange,maturity,delta,market_price,model_price\n"
            "2024-01-01T00:00:00Z,deribit,0.01,0.25,1200,1180\n"
            "2024-01-01T00:01:00Z,okx,,0.45,980,920\n"
        ),
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/deviation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["n_rows"] == 2
    # observed=False keeps every (expiry, delta) combo in the records; only
    # judge the populated cells (value is not null).
    populated = [
        row
        for row in payload["heatmap_records"]
        if row["abs_deviation_bps"] is not None
    ]
    # 0.01y = 3.65 days -> "<=7D"; |delta| = 0.25 -> "10-25d".
    assert any(
        row["expiry_bucket"] == "<=7D" and row["delta_bucket"] == "10-25d"
        for row in populated
    )
    # The row with missing maturity must NOT be merged into "<=7D": it keeps
    # its own "UNKNOWN" expiry bucket (|delta| = 0.45 -> "40-60d").
    assert any(
        row["expiry_bucket"] == "UNKNOWN" and row["delta_bucket"] == "40-60d"
        for row in populated
    )
    assert not any(
        row["expiry_bucket"] == "<=7D" and row["delta_bucket"] == "40-60d"
        for row in populated
    )


def test_dashboard_deviation_api_emits_null_for_empty_heatmap_cells(tmp_path):
    # Regression: unpopulated (expiry, delta) combinations were reported as
    # 0.0 bps — indistinguishable from perfectly matched prices; they are
    # now emitted as null.
    csv_path = tmp_path / "options_deviation.csv"
    csv_path.write_text(
        (
            "timestamp,exchange,maturity,delta,market_price,model_price\n"
            "2024-01-01T00:00:00Z,deribit,0.02,0.25,1200,1180\n"
        ),
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/deviation")

    assert response.status_code == 200
    payload = response.json()
    # The response must be strict-JSON safe: no NaN floats anywhere.
    assert "NaN" not in response.text
    populated = [
        row for row in payload["heatmap_records"] if row["abs_deviation_bps"] is not None
    ]
    assert populated, "populated cell must still be reported"
    assert all(row["abs_deviation_bps"] > 0.0 for row in populated)
    empty_records = [
        row for row in payload["heatmap_records"] if row["abs_deviation_bps"] is None
    ]
    assert empty_records, "empty cells must serialize as null, not 0.0"
    empty_cells = [
        value
        for column in payload["heatmap_matrix"].values()
        for value in column.values()
        if value is None
    ]
    assert empty_cells, "empty cells must serialize as null, not 0.0"


def test_dashboard_lists_files_and_renders_html(tmp_path):
    csv_path = tmp_path / "backtest.csv"
    csv_path.write_text(
        "timestamp,equity\n" "2024-01-01T00:00:00Z,100\n" "2024-01-01T00:01:00Z,101\n",
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        files_response = client.get("/api/files")
        html_response = client.get("/")

    assert files_response.status_code == 200
    assert files_response.json()["files"] == ["backtest.csv"]
    assert html_response.status_code == 200
    assert "CORP Research Dashboard" in html_response.text
    assert "Return Distribution" in html_response.text


def test_dashboard_missing_files_returns_404(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 404


def test_dashboard_deviation_api_returns_heatmap_and_alerts(tmp_path):
    csv_path = tmp_path / "options_deviation.csv"
    csv_path.write_text(
        (
            "timestamp,exchange,maturity,delta,market_price,model_price\n"
            "2024-01-01T00:00:00Z,deribit,0.02,0.25,1200,1180\n"
            "2024-01-01T00:01:00Z,okx,0.08,0.45,980,920\n"
            "2024-01-01T00:02:00Z,deribit,0.20,0.15,760,700\n"
        ),
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/deviation", params={"threshold_bps": 400})

    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert "heatmap_records" in payload
    assert "alerts" in payload
    assert payload["summary"]["n_rows"] == 3
    assert payload["summary"]["n_alerts"] >= 1


def test_dashboard_live_deviation_api_aligns_cex_defi_sources(tmp_path):
    cex_path = tmp_path / "cex_quotes.csv"
    cex_path.write_text(
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:00Z,BTC-OPT,call,0.05,0.25,1200,deribit\n"
            "2024-01-01T00:01:00Z,BTC-OPT,call,0.05,0.25,1180,okx\n"
        ),
        encoding="utf-8",
    )

    defi_path = tmp_path / "defi_quotes.csv"
    defi_path.write_text(
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:00:20Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
            "2024-01-01T00:01:40Z,BTC-OPT,call,0.05,0.25,1130,ribbon\n"
        ),
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get(
            "/api/deviation/live",
            params={
                "threshold_bps": 200.0,
                "cex_file": str(cex_path),
                "defi_file": str(defi_path),
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"]["rows_aligned"] == 2
    assert payload["summary"]["n_rows"] == 2
    assert payload["summary"]["n_alerts"] >= 1
    assert payload["sources"]["mode"] == "file"


def test_dashboard_live_deviation_api_honors_alignment_tolerance(tmp_path):
    cex_path = tmp_path / "cex_quotes.csv"
    cex_path.write_text(
        (
            "timestamp,symbol,option_type,maturity,delta,price,exchange\n"
            "2024-01-01T00:00:59Z,BTC-OPT,call,0.05,0.25,1200,okx\n"
        ),
        encoding="utf-8",
    )
    defi_path = tmp_path / "defi_quotes.csv"
    defi_path.write_text(
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:01:01Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
        encoding="utf-8",
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        fail_response = client.get(
            "/api/deviation/live",
            params={
                "threshold_bps": 200.0,
                "cex_file": str(cex_path),
                "defi_file": str(defi_path),
                "align_tolerance_seconds": 1,
            },
        )
        pass_response = client.get(
            "/api/deviation/live",
            params={
                "threshold_bps": 200.0,
                "cex_file": str(cex_path),
                "defi_file": str(defi_path),
                "align_tolerance_seconds": 3,
            },
        )

    assert fail_response.status_code == 422
    assert pass_response.status_code == 200
    assert pass_response.json()["summary"]["n_rows"] == 1


def test_dashboard_live_deviation_api_supports_provider_mode(tmp_path, monkeypatch):
    defi_path = tmp_path / "defi_quotes.csv"
    defi_path.write_text(
        (
            "timestamp,symbol,option_type,maturity,delta,price,source\n"
            "2024-01-01T00:00:20Z,BTC-OPT,call,0.05,0.25,1140,lyra\n"
        ),
        encoding="utf-8",
    )
    mock_dataset = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "BTC-OPT",
                "option_type": "call",
                "maturity": 0.05,
                "delta": 0.25,
                "market_price": 1200.0,
                "model_price": 1140.0,
                "venue": "cex_vs_defi",
            },
            {
                "timestamp": "2024-01-01T00:01:00Z",
                "symbol": "BTC-OPT",
                "option_type": "call",
                "maturity": 0.05,
                "delta": 0.25,
                "market_price": 1180.0,
                "model_price": 1130.0,
                "venue": "cex_vs_defi",
            },
        ]
    )
    fetch_mock = AsyncMock(return_value=mock_dataset)
    monkeypatch.setattr(
        "execution.research_dashboard.build_cex_defi_deviation_dataset_live",
        fetch_mock,
    )

    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get(
            "/api/deviation/live",
            params={
                "threshold_bps": 200.0,
                "cex_provider": "okx",
                "defi_file": str(defi_path),
                "underlying": "BTC-USD",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"]["mode"] == "provider"
    assert payload["sources"]["cex_provider"] == "okx"
    assert payload["sources"]["rows_aligned"] == 2
    fetch_mock.assert_awaited_once()


def test_dashboard_live_deviation_api_requires_sources(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/deviation/live")

    assert response.status_code == 422
