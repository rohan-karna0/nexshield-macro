---
title: NexShield Production Portfolio Plan
author: NexShield Team
date: May 2026
---

# NexShield Production Portfolio Plan

**General-purpose fraud prevention platform**  
**Phase 1 focus:** Macro Fraud Wave Predictor  
**Timeline:** 4–6 weeks (full platform) | 2.5–3 weeks (macro phase only)

---

## Document purpose

This plan describes how to build NexShield as a **bank-agnostic**, production-grade fraud platform suitable for portfolio and interviews at top technology firms and financial institutions (JPMorgan, Morgan Stanley, Tower Research, DE Shaw, etc.). It is **not** tied to SBI, YONO, or any single bank.

---

## 1. Executive summary

NexShield prevents fraud at two levels:

1. **Macro layer** — Predicts *when* and *where* regional fraud campaigns will spike (time-series + geospatial clustering).
2. **Edge layer** (later) — On-device behavioral anomaly detection with federated learning; no raw PII leaves the client.

**Current scope:** Complete the **Macro Fraud Wave Predictor** first (data, models, API, dashboard, benchmarks). Edge SDK, risk bridge, and federated learning are deferred until macro is validated.

### ML strategy: train from scratch or fine-tune?

| Layer | Right approach | Avoid |
|-------|----------------|--------|
| **Fraud Wave Predictor** | LightGBM + Prophet on daily region aggregates | LLM fine-tuning; huge deep nets without baselines |
| **Geospatial risk** | HDBSCAN / DBSCAN on event coordinates | Neural clustering |
| **On-device anomaly** (later) | Tiny autoencoder/CNN → TFLite INT8 | Sending behavior to cloud |
| **Federated learning** (later) | FedAvg via Flower (simulated clients) | Fake production-scale claims |
| **Scam text** (stretch) | Fine-tune DistilBERT | Pretrain BERT from scratch |

**Do not train foundation models from scratch** for this project.

---

## 2. Product positioning

### Design principles

- **Tenant model** — Each deployer gets `tenant_id`, config (geo level, alerts, thresholds), isolated data.
- **Pluggable ingestion** — CSV, webhook, public dataset adapters (not one national API).
- **Configurable geography** — `region_id` = postal code, city, state, or grid; any country.
- **Edge SDK** — Integrates with any mobile/web app via REST + local TFLite.
- **Compliance** — Document PCI-DSS, GDPR, SOC2; regional laws as deployment notes.

### Terminology

| Original (hackathon) | General term |
|----------------------|--------------|
| District | `region_id` |
| Bank-specific reports | `fraud_signal_events` |
| In-app on-device | `edge_client` / `device_sdk` |
| National heatmap | `geo_risk_map` |

---

## 3. Phase 1: Macro Fraud Wave Predictor (detailed)

### 3.1 What you predict

| Question | Output |
|----------|--------|
| **When** will fraud signals spike? | 7-day forecast per `region_id` |
| **Where** are campaigns clustering? | HDBSCAN clusters on last 14 days |
| **How urgent?** | `risk_score` 0–100 + alert tier |

**Target variable (per tenant, region, day):**

```
y_t = count(fraud_signal_events in region r on day t)
```

**Alert logic:**

```
forecast_{t+1..t+7} = model.predict(horizon=7)
IF forecast > percentile_95_historical OR velocity_7d > 2.0:
    EMIT CAMPAIGN_WARNING(region r)
```

---

### 3.2 Dataset options

Macro models need **aggregates by (date, region)** with optional lat/lon.

#### Tier A — Start here

| Dataset | Macro use | Pros | Cons |
|---------|-----------|------|------|
| [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) | Daily fraud counts by engineered `region_id` | Fast, classic | No real geography |
| [IEEE-CIS Fraud](https://www.kaggle.com/competitions/ieee-fraud-detection) | Daily fraud per `addr1` / product proxy | High interview credibility | Large; weak geo |
| [Mendeley Synthetic Banking 1M](https://data.mendeley.com/datasets/ktbthg777x/1) | Geo bucket daily counts | Built-in geo anomalies | Synthetic only |
| **Your synthetic generator** | Inject 2–3 campaigns/week | Proves **lead time** metric | Must label as synthetic |

**Recommended:** Mendeley or synthetic for **map + campaigns** + IEEE-CIS subsample for **benchmark slide**.

#### Tier B — Supplements

| Dataset | Notes |
|---------|-------|
| [Nigerian FinTx 5M](https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset) | 20 cities → city-day aggregates |
| [FTC Consumer Sentinel](https://www.ftc.gov/exploredata) | Real US complaints by state/month |
| ECB / open card-fraud stats | Country-level exogenous features only |

#### Tier C — Do not use as primary

- Ethereum / GNN address graphs — wrong granularity  
- 10-K corporate fraud text — wrong problem  
- Credit-card-only sets (ULB, etc.) — no geography  

#### Honest limitations

You will **not** have live bank complaint feeds or national cybercrime APIs in a portfolio build. Use adapters + public/synthetic data.

---

### 3.3 Data management

#### Canonical schema

**Bronze — `fraud_signal_event`**

```yaml
event_id: uuid
tenant_id: string
timestamp: datetime_utc
region_id: string          # e.g. US-CA-SF, NG-LAG
lat: float | null
lon: float | null
signal_type: enum          # transaction_fraud | complaint | chargeback | report
amount_usd: float | null
fraud_label: bool | null
source: string             # paysim | ieee_cis | synthetic | ftc
metadata: json
```

**Gold — `region_daily_aggregate`**

```yaml
tenant_id, region_id, date,
event_count, fraud_count, amount_sum,
lag_1d, lag_7d, lag_14d, lag_28d,
rolling_mean_7d, rolling_std_7d,
dow, month, is_holiday,
peer_zscore
```

#### Storage layout

```
data/
  raw/{source}/{download_date}/     # gitignored
  bronze/events.parquet
  gold/region_daily/{tenant}/
```

#### Orchestration

- **Week 1–2:** `Makefile` + Python scripts  
- **Optional:** Prefect/Dagster, DVC for parquet hashes, Great Expectations (5 checks)

---

### 3.4 Model selection

Problem type: **sparse count time series per region** with sudden campaign spikes.

| Model | Verdict |
|-------|---------|
| Naive / seasonal naive | **Required baseline** |
| Holt-Winters / ETS | Strong baseline |
| **Prophet** | Interpretability (top ~20 regions by volume) |
| **LightGBM on lags** | **Champion** — best accuracy at scale |
| SARIMAX | Optional if Prophet underperforms |
| Small LSTM/TCN | Only if >5% MAPE better than LightGBM |
| Chronos / TimesFM | Ablation only |
| LLM | **Do not use** |

#### Production ensemble

```
risk_score = 0.4 × forecast_score
           + 0.4 × anomaly_spike_score
           + 0.2 × geo_cluster_score
```

**Branches:**

1. **Forecast** — Global LightGBM: `region_id` (target encoded), lags, rolling stats, calendar, `peer_zscore`; multi-horizon h=1..7  
2. **Explainability** — Prophet on top-N regions for dashboard tooltips  
3. **Geospatial** — HDBSCAN on last 14 days of lat/lon events → cluster growth rate  

**Champion rule:** Rolling-origin backtest; lowest MAPE at h=1 and h=7; precision@alert ≥ 0.5 on synthetic campaigns.

---

### 3.5 Evaluation methodology

**Never** use random train/test split on time series.

**Rolling-origin backtest:**

```
FOR each cutoff T in validation window:
    TRAIN on [T - 365 days, T]
    PREDICT T+1 .. T+7
    COMPARE to actual
```

| Metric | Purpose |
|--------|---------|
| MAPE / sMAPE | Forecast accuracy |
| RMSE on counts | Penalize spike misses |
| Precision / Recall @ alert | Campaign detection |
| Lead time | Alert ≥24h before peak (synthetic) |
| False alerts / region / month | Ops cost |

Report separately for **high-volume** vs **low-volume** regions.

---

### 3.6 Optimization strategies

#### A. ML optimization

1. **Optuna** — 50–100 trials on LightGBM (`num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`); objective = mean sMAPE on rolling backtest  
2. **Target transform** — `log1p(count)` or `objective='poisson'` in LightGBM  
3. **Alert threshold** — Tune on precision–recall curve (campaigns are rare)  
4. **Features** — `velocity_7d`, `peer_zscore`, Fourier weekly terms, holidays per tenant country  
5. **Cold-start regions** — National Prophet forecast × population weight  

#### B. Engineering optimization

| Area | Strategy |
|------|----------|
| Training | Nightly global LightGBM; incremental daily append |
| Inference | Precompute 7-day forecasts → Redis/Postgres; API <5ms |
| Batch | Micro-batch aggregates every 15 min; forecast refresh hourly |
| Database | Postgres + PostGIS; index `(tenant_id, region_id, date)` |
| Caching | `risk_score` TTL 1h; invalidate on ingestion spike |
| Registry | MLflow: `champion`, `mape_h7`, `git_sha` |

#### C. Operational optimization

- Per-tenant thresholds in config (not hardcoded)  
- Alert deduplication: max 1 CAMPAIGN_WARNING per region per 24h  
- Explainability payload: top 3 features + forecast delta on every alert  
- **Shadow mode** for new models before promotion  

---

### 3.7 Macro service architecture

```
Ingestion adapters → bronze events → daily aggregate → gold
                              ↓
         Rolling backtest → LightGBM (champion) + Prophet (explain)
                              ↓
         HDBSCAN clusters ──────────→ fuse risk_score → API + cache
                              ↓
                        Streamlit geo dashboard
```

#### API endpoints (v1)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/events:batch` | Ingest fraud_signal_events |
| GET | `/v1/tenants/{tenant}/regions/{id}/risk` | Current score + tier |
| GET | `/v1/tenants/{tenant}/regions/{id}/forecast?horizon=7` | Forecast series |
| GET | `/v1/tenants/{tenant}/clusters?since=14d` | Geo clusters |
| GET | `/v1/tenants/{tenant}/alerts` | Paginated alert history |

---

### 3.8 Macro-only timeline (~2.5–3 weeks)

| Phase | Days | Deliverable |
|-------|------|-------------|
| **0 — Data** | 3–4 | Schema, 2 adapters, gold aggregates, data quality checks |
| **1 — Baselines** | 3–4 | Rolling backtest; naive, Prophet, LightGBM leaderboard |
| **2 — Geo fusion** | 2–3 | HDBSCAN, fused `risk_score`, alert rules |
| **3 — Serve** | 3–4 | FastAPI, MLflow, Streamlit map, README metrics |

#### Definition of done (macro)

- [ ] `make macro-train && make macro-serve` works in Docker  
- [ ] README backtest table: baselines vs champion  
- [ ] Streamlit map: `risk_score` + forecast on region click  
- [ ] Synthetic demo: ≥24h lead time before campaign peak  

#### Folder structure (macro slice)

```
services/
  ingestion/
  macro-predictor/
    etl/aggregate.py
    train/backtest.py
    train/lightgbm_model.py
    train/prophet_explain.py
    geo/hdbscan_clusters.py
    serve/api.py
    scoring/fuse_risk.py
adapters/
  synthetic_campaign.py
  ieee_cis_macro.py
  mendeley_banking_macro.py
ml/benchmarks/macro_leaderboard.json
frontend/macro_dashboard/    # Streamlit
```

---

### 3.9 Interview talking points (macro)

- **Why not LSTM first?** Sparse counts across hundreds of regions; global LightGBM wins with less ops complexity.  
- **Why LightGBM + Prophet?** Accuracy at scale + human-readable seasonality for ops.  
- **Why synthetic data?** Public txn data lacks labeled campaigns; inject waves to measure lead time, validate on IEEE-CIS aggregates.

---

## 4. Full platform (phases 2+)

### 4.1 Components (after macro)

| Component | Technology | Purpose |
|-----------|------------|---------|
| Risk-Context Bridge | Encrypted signals, policy engine | Raise edge sensitivity when macro risk rises |
| Edge SDK | TFLite INT8, <500KB | On-device behavioral anomaly + block |
| Federated learning | Flower FedAvg | Improve global model without raw data export |
| Alert dispatcher | Webhook / email / push stubs | Notify ops and clients |

### 4.2 Tech stack

Python 3.11 · FastAPI · PostgreSQL + PostGIS · Redis (optional) · Streamlit or React · scikit-learn · LightGBM · Prophet · TensorFlow → TFLite · Flower · Docker · GitHub Actions · MLflow

### 4.3 Six-week timeline (summary)

| Week | Focus |
|------|-------|
| 1 | Monorepo, tenant config, pipelines, baselines |
| 2 | Macro predictor service + API |
| 3 | TFLite edge anomaly + simulator |
| 4 | Risk bridge + E2E test |
| 5 | Federated learning simulation |
| 6 | CI/CD, observability, security docs, demo video |

### 4.4 Target metrics

| Metric | Target |
|--------|--------|
| Region forecast MAPE | <15% on synthetic holdout |
| Campaign lead time | ≥24h before peak |
| On-device inference | <10ms p99 (TFLite INT8) |
| FL uplift | +5–10% F1 vs local-only |
| API availability | 99.9% in local load test |

---

## 5. Positioning for top firms

| Firm type | Emphasize |
|-----------|-----------|
| **Global banks** | ROI, explainable alerts, audit trail, multi-tenant, compliance |
| **Quant firms** | Rolling backtest, no leakage, latency budgets, baselines |
| **Big tech / platform** | CI/CD, model registry, federated + on-device constraints |
| **Consulting** | Phased rollout, KPI impact, ops dashboard |

**One-liner:**  
*NexShield is a deployable fraud platform: macro campaign forecasting drives regional risk policy, while a federated edge SDK catches personalized anomalies—no PII leaves the client, any institution plugs in via adapters.*

---

## 6. Scope boundaries (be honest)

- No live core-banking integration in portfolio timeline  
- No claim of nation-scale federated learning (simulate 10–50 clients)  
- No SBI/YONO branding in repo or demo  
- Docker Compose over Kubernetes for v1  

---

## 7. Success criteria (full project)

1. `docker compose up` runs ingest → macro → bridge → edge alert  
2. Public GitHub with README, benchmarks, demo video  
3. `make train-all` regenerates models and metrics  
4. Can whiteboard privacy, evaluation, and model choices in 10 minutes  

---

*End of document*
