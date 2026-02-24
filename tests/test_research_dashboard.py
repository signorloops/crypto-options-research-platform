"""Tests for the research dashboard web app."""

from fastapi.testclient import TestClient

from execution.research_dashboard import create_dashboard_app


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


def test_dashboard_live_deviation_api_requires_sources(tmp_path):
    app = create_dashboard_app(results_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/deviation/live")

    assert response.status_code == 422
