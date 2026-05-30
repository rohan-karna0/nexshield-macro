#!/usr/bin/env python3
"""
End-to-end macro pipeline demo.

Run from repo root:
    pip install -r requirements.txt
    python scripts/run_macro_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MACRO_DIR = ROOT / "services" / "macro-predictor"

# Repo root: nexshield, adapters, services.ingestion
sys.path.insert(0, str(ROOT))
# macro-predictor: etl.*, train.* (folder name has hyphen, not importable as package)
sys.path.insert(0, str(MACRO_DIR))

from adapters.synthetic_campaign import generate_synthetic_events
from etl.aggregate import aggregate_and_save
from nexshield.config import BENCHMARK_DIR, DEFAULT_TENANT
from services.ingestion.pipeline import ingest_from_adapter
from services.ingestion.store import load_bronze
from train.backtest import run_naive_backtest
from train.lightgbm_model import backtest_lightgbm, feature_importance_report, train_lightgbm
from train.prophet_explain import backtest_prophet_top_regions


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    tenant = DEFAULT_TENANT

    _banner("STEP 1 — INGEST synthetic fraud events")
    ingest_result = ingest_from_adapter(
        generate_synthetic_events,
        tenant,
        replace=True,
        num_days=120,
        seed=42,
    )
    print(f"  Tenant:           {ingest_result['tenant_id']}")
    print(f"  Events generated: {ingest_result['events_ingested']}")
    print(f"  Bronze rows:      {ingest_result['events_written']}")

    bronze = load_bronze(tenant)
    print(f"  Date range:       {bronze['timestamp'].min()} -> {bronze['timestamp'].max()}")
    print(f"  Regions:          {bronze['region_id'].nunique()}")

    _banner("STEP 2 — ETL: bronze -> gold region_daily")
    gold = aggregate_and_save(bronze, tenant)
    print(f"  Gold rows:        {len(gold)}  (regions x days)")
    print(f"  Columns:          {list(gold.columns)}")
    print("\n  Sample (last 3 rows for US-NY-NYC):")
    sample = gold[gold["region_id"] == "US-NY-NYC"].tail(3)
    print(sample[["date", "event_count", "lag_7d", "velocity_7d", "peer_zscore"]].to_string(index=False))

    _banner("STEP 3 — BACKTEST models (rolling-origin)")
    print("  Evaluating naive baseline...")
    naive_metrics = run_naive_backtest(gold)
    print(f"  Naive   MAPE h=1: {naive_metrics.mape_h1:.2f}%  |  MAPE h=7: {naive_metrics.mape_h7:.2f}%")

    print("\n  Evaluating macro model (LightGBM or sklearn fallback)...")
    lgbm_metrics, _ = backtest_lightgbm(gold)
    print(f"  LGBM    MAPE h=1: {lgbm_metrics.mape_h1:.2f}%  |  MAPE h=7: {lgbm_metrics.mape_h7:.2f}%")

    print("\n  Evaluating Prophet (top 5 regions)...")
    prophet_metrics, _, explanations = backtest_prophet_top_regions(gold, top_n=5)
    ph1 = prophet_metrics.mape_h1
    ph7 = prophet_metrics.mape_h7
    print(f"  Prophet MAPE h=1: {ph1:.2f}%  |  MAPE h=7: {ph7:.2f}%")

    _banner("STEP 4 — TRAIN final macro model on all data")
    final_model, _encoder, feature_names, backend = train_lightgbm(gold)
    print(f"  Backend:          {backend}")
    importance = feature_importance_report(final_model, feature_names)
    print("  Top feature importances:")
    print(importance.to_string(index=False))

    _banner("STEP 5 — PROPHET explanations (top regions)")
    if explanations:
        for ex in explanations[:3]:
            print(f"\n  Region: {ex.region_id}")
            print(f"    Trend component:   {ex.trend:+.2f}")
            print(f"    Weekly component:  {ex.weekly:+.2f}")
            print(f"    Next 7d forecast:  {[round(v, 1) for v in ex.forecast_next_7d]}")
    else:
        print("  (Prophet skipped or insufficient data)")

    _banner("STEP 6 — LEADERBOARD saved")
    leaderboard = {
        "tenant_id": tenant,
        "models": [
            naive_metrics.to_dict(),
            lgbm_metrics.to_dict(),
            prophet_metrics.to_dict(),
        ],
        "feature_importance": importance.to_dict(orient="records"),
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BENCHMARK_DIR / "macro_leaderboard.json"
    out_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    print(f"  Written to: {out_path}")

    print("\n  SUMMARY")
    models = [m for m in leaderboard["models"] if m.get("mape_h7") == m.get("mape_h7")]
    if models:
        best = min(models, key=lambda m: m["mape_h7"])
        print(f"  Best MAPE@h=7: {best['model_name']} ({best['mape_h7']}%)")


if __name__ == "__main__":
    main()
