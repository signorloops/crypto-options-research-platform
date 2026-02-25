# Research Dashboard

CORP includes a lightweight web dashboard for interactive result analysis using Plotly.

## Start Dashboard

```bash
python -m execution.research_dashboard
```

Environment variables:

- `DASHBOARD_HOST` (default: `0.0.0.0`)
- `DASHBOARD_PORT` (default: `8501`)
- `CORP_RESULTS_DIR` (default: `results`)
- `CEX_QUOTES_FILE` (optional: live CEX quote file path for cross-market monitor)
- `CEX_QUOTES_PROVIDER` (optional: live CEX provider, e.g. `okx`)
- `DEFI_QUOTES_FILE` (optional: live DeFi quote file path for cross-market monitor)

## Features

- Auto-discover `.csv` and `.parquet` result files
- Interactive time-series chart for the primary metric
- Return distribution histogram
- Summary statistics table
- Cross-market deviation heatmap (expiry bucket × delta bucket)
- Threshold-based deviation alerts (default: `300 bps`)

## Endpoints

- `GET /` web dashboard UI
- `GET /api/files` list available result files
- `GET /api/deviation` compute cross-market/model deviation report
- `GET /api/deviation/live` compute aligned CEX-vs-DeFi deviation from file/provider sources
- `GET /health` dashboard process health

`/api/deviation` query params:

- `file`: optional result filename (default latest file)
- `threshold_bps`: alert threshold in bps (default `300`)

`/api/deviation/live` query params:

- `cex_file`: CEX quote file path (optional if `CEX_QUOTES_FILE` is set)
- `cex_provider`: CEX live provider (optional if `cex_file` is set, currently supports `okx`)
- `defi_file`: DeFi quote file path (required unless `DEFI_QUOTES_FILE` is set)
- `underlying`: provider mode underlying (default `BTC-USD`)
- `align_tolerance_seconds`: fallback nearest-timestamp alignment tolerance (default `60`)
- `threshold_bps`: alert threshold in bps (default `300`)

Live endpoint behavior:

- Normalizes quote schemas from CEX/DeFi sources
- Aligns rows by minute key first; if no exact match, uses nearest timestamp fallback within tolerance
- Produces deviation report with the same shape as `/api/deviation`

Minimal columns for deviation analysis:

- Market quote: one of `market_price`, `quote_price`, `option_price`, `price`
- Model quote: one of `model_price`, `theoretical_price`, `benchmark_price`, `fair_value`

Optional columns to improve bucket quality:

- Expiry years: `expiry_years` / `maturity` / `time_to_expiry` / `tau`
- Delta: `delta` / `abs_delta` / `delta_abs`
- Venue: `venue` / `exchange` / `source` / `market`

Recommended live source columns:

- `timestamp`
- `symbol` or `instrument`
- `option_type`
- `maturity`/`expiry_years`
- `delta`
- `price`
- venue label (`exchange`/`source`)

CLI snapshot automation:

```bash
make live-deviation-snapshot
```

This generates:

- `artifacts/live-deviation-snapshot.md`
- `artifacts/live-deviation-snapshot.json`

## Intended Usage

- Strategy research iteration
- Quick sanity checks after backtests
- Interactive sharing of experiment outcomes
