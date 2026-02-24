"""
Tests for model-zoo benchmark script fixture I/O.
"""

from pathlib import Path

import numpy as np

from validation_scripts.pricing_model_zoo_benchmark import (
    _build_synthetic_quotes,
    load_quotes_json,
    run_benchmark,
    save_quotes_json,
)


def test_quotes_json_roundtrip(tmp_path):
    quotes = _build_synthetic_quotes(
        spot=50000.0,
        rate=0.02,
        sigma=0.60,
        seed=42,
        n_per_bucket=1,
    )
    output_path = tmp_path / "quotes.json"
    save_quotes_json(str(output_path), quotes)

    loaded = load_quotes_json(str(output_path))
    assert len(loaded) == len(quotes)
    assert loaded[0].spot == quotes[0].spot
    assert loaded[0].strike == quotes[0].strike
    assert np.isclose(loaded[0].maturity, quotes[0].maturity, atol=1e-8)
    assert np.isclose(loaded[0].market_price, quotes[0].market_price, atol=1e-6)


def test_benchmark_ranking_stable_with_fixed_quotes(tmp_path):
    quotes = _build_synthetic_quotes(
        spot=50000.0,
        rate=0.02,
        sigma=0.60,
        seed=7,
        n_per_bucket=1,
    )
    output_path = tmp_path / "quotes.json"
    save_quotes_json(str(output_path), quotes)
    loaded = load_quotes_json(str(output_path))

    table_a = run_benchmark(quotes=quotes)
    table_b = run_benchmark(quotes=loaded)

    assert table_a["model"].tolist() == table_b["model"].tolist()
    assert np.allclose(table_a["rmse"].to_numpy(), table_b["rmse"].to_numpy())
    assert np.allclose(table_a["mae"].to_numpy(), table_b["mae"].to_numpy())


def test_tracked_fixture_file_runs_benchmark():
    fixture_path = Path("validation_scripts/fixtures/model_zoo_quotes_seed42.json")
    quotes = load_quotes_json(str(fixture_path))
    table = run_benchmark(quotes=quotes)

    assert len(quotes) == 20
    assert not table.empty
    assert table.iloc[0]["model"] == "bates"
