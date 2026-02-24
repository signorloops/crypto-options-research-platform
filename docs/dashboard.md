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
- `GET /api/deviation/live` compute aligned CEX-vs-DeFi deviation from two quote sources
- `GET /health` dashboard process health

`/api/deviation` query params:

- `file`: optional result filename (default latest file)
- `threshold_bps`: alert threshold in bps (default `300`)

`/api/deviation/live` query params:

- `cex_file`: CEX quote file path (optional if `CEX_QUOTES_FILE` is set)
- `defi_file`: DeFi quote file path (optional if `DEFI_QUOTES_FILE` is set)
- `threshold_bps`: alert threshold in bps (default `300`)

Live endpoint behavior:

- Normalizes quote schemas from CEX/DeFi sources
- Aligns rows by minute timestamp + symbol + option type + expiry bucket + delta bucket
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

## Intended Usage

- Strategy research iteration
- Quick sanity checks after backtests
- Interactive sharing of experiment outcomes
