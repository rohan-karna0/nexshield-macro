# NexShield — Macro Fraud Wave Predictor (Phase 1)

Bank-agnostic fraud **campaign forecasting**: predict when/where regional fraud signal volume will spike.

## Quick start

```bash
cd NexShield
pip install -r requirements.txt
python scripts/run_macro_pipeline.py
```

You should see 6 printed steps: ingest → ETL → backtest → train → Prophet explain → JSON leaderboard.

Results are saved to `ml/benchmarks/macro_leaderboard.json`.

## Project layout

```
adapters/
  synthetic_campaign.py     # Generates demo fraud events with injected campaigns

services/
  ingestion/
    store.py                # Bronze Parquet read/write
    pipeline.py             # Adapter -> bronze orchestration
    api.py                  # FastAPI batch ingest (optional)

  macro-predictor/
    etl/
      aggregate.py          # Bronze -> gold region_daily + lag features
    train/
      backtest.py           # Rolling-origin evaluation (no data leakage)
      lightgbm_model.py     # Champion forecaster
      prophet_explain.py    # Interpretable seasonality for top regions

scripts/
  run_macro_pipeline.py     # End-to-end demo
```

## Code walkthrough

### 1. Ingestion (`services/ingestion/`)

| File | Role |
|------|------|
| `store.py` | Persists `FraudSignalEvent` rows to `data/bronze/{tenant}_events.parquet` |
| `pipeline.py` | Calls an adapter function and writes bronze (replace or append) |
| `api.py` | `POST /v1/events:batch` for HTTP ingest |

### 2. ETL (`macro-predictor/etl/aggregate.py`)

Raw events are **counted per (region, day)** then enriched with:

- **Lags** (`lag_1d`, `lag_7d`, …) — past volume
- **velocity_7d** — last 7 days vs previous 7 (campaign detector)
- **peer_zscore** — how hot this region is vs national average that day

Output: `data/gold/region_daily/{tenant}/region_daily.parquet`

### 3. Backtest (`train/backtest.py`)

Uses **rolling-origin** splits (never random shuffle):

```
train on [T-90, T]  ->  predict T+1 .. T+7  ->  score vs actual
```

Metrics: MAPE and RMSE at horizon 1 and 7.

### 4. LightGBM (`train/lightgbm_model.py`)

One **global** model across all regions with target-encoded `region_id`. Trains on `log1p(event_count)` to handle spikes.

### 5. Prophet (`train/prophet_explain.py`)

Fits on **top 5 regions** only — slower but gives human-readable trend/weekly components for dashboards.

## Start ingestion API (optional)

```bash
uvicorn services.ingestion.api:app --reload --port 8000
```

## Data flow

```
synthetic adapter -> bronze parquet -> gold aggregates -> backtest -> train -> benchmarks
```
