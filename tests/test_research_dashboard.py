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
