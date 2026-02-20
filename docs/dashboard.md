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

## Features

- Auto-discover `.csv` and `.parquet` result files
- Interactive time-series chart for the primary metric
- Return distribution histogram
- Summary statistics table

## Endpoints

- `GET /` web dashboard UI
- `GET /api/files` list available result files
- `GET /health` dashboard process health

## Intended Usage

- Strategy research iteration
- Quick sanity checks after backtests
- Interactive sharing of experiment outcomes
