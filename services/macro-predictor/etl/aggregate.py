"""
Gold-layer ETL: bronze events → region_daily_aggregate.

The macro model does NOT train on raw events. It trains on daily counts
per region, enriched with lags and rolling statistics (classic time-series
features for tree models like LightGBM).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nexshield.config import GOLD_DIR


def gold_path(tenant_id: str) -> Path:
    """Parquet path for gold region-daily aggregates."""
    tenant_dir = GOLD_DIR / tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return tenant_dir / "region_daily.parquet"


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week and month — fraud often has weekly seasonality."""
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    # Simple US holiday proxy: Jan 1, Jul 4, Dec 25 (extend per tenant)
    df["is_holiday"] = (
        ((df["date"].dt.month == 1) & (df["date"].dt.day == 1))
        | ((df["date"].dt.month == 7) & (df["date"].dt.day == 4))
        | ((df["date"].dt.month == 12) & (df["date"].dt.day == 25))
    ).astype(int)
    return df


def _add_lag_features_vectorized(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Per-region lag and rolling features using vectorized groupby shifts.

    All shifts use PAST data only (no leakage into the future).
    """
    daily = daily.sort_values(["region_id", "date"]).copy()
    g = daily.groupby("region_id")["event_count"]

    for lag in (1, 7, 14, 28):
        daily[f"lag_{lag}d"] = g.shift(lag)

    shifted = g.shift(1)
    daily["rolling_mean_7d"] = shifted.transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    daily["rolling_std_7d"] = shifted.transform(lambda s: s.rolling(window=7, min_periods=2).std()).fillna(0.0)

    last7 = shifted.transform(lambda s: s.rolling(7, min_periods=1).sum())
    prev7 = g.shift(8).transform(lambda s: s.rolling(7, min_periods=1).sum())
    daily["velocity_7d"] = (last7 / prev7.replace(0, np.nan)).fillna(1.0)

    return daily


def build_region_daily(bronze: pd.DataFrame, tenant_id: str) -> pd.DataFrame:
    """
    Transform bronze events into gold region_daily_aggregate.

    Steps
    -----
    1. Filter to tenant
    2. Truncate timestamps to UTC date
    3. Groupby (region_id, date) → counts and amount sums
    4. Reindex so missing days get event_count=0 (important for forecasting)
    5. Add lags, rolling stats, calendar, peer_zscore
    """
    if bronze.empty:
        return pd.DataFrame()

    df = bronze.copy()
    df = df[df["tenant_id"] == tenant_id]
    df["date"] = df["timestamp"].dt.floor("D")

    daily = (
        df.groupby(["tenant_id", "region_id", "date"], as_index=False)
        .agg(
            event_count=("event_id", "count"),
            fraud_count=("fraud_label", lambda s: int(s.fillna(False).sum())),
            amount_sum=("amount_usd", lambda s: float(s.fillna(0).sum())),
        )
    )

    # Fill missing (region, day) pairs with zeros
    all_regions = daily["region_id"].unique()
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D", tz="UTC")
    full_index = pd.MultiIndex.from_product(
        [[tenant_id], all_regions, date_range],
        names=["tenant_id", "region_id", "date"],
    )
    daily = (
        daily.set_index(["tenant_id", "region_id", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    for col in ("event_count", "fraud_count"):
        daily[col] = daily[col].astype(int)
    daily["amount_sum"] = daily["amount_sum"].astype(float)

    # Lags per region (vectorized — avoids pandas groupby.apply column loss)
    daily = _add_lag_features_vectorized(daily)
    daily = _add_calendar_features(daily)

    # Peer z-score: how unusual is this region vs all regions that day?
    national = daily.groupby("date")["event_count"].agg(["mean", "std"]).rename(
        columns={"mean": "national_mean", "std": "national_std"}
    )
    daily = daily.merge(national, on="date", how="left")
    daily["national_std"] = daily["national_std"].replace(0, np.nan)
    daily["peer_zscore"] = (
        (daily["event_count"] - daily["national_mean"]) / daily["national_std"]
    ).fillna(0.0)

    daily = daily.drop(columns=["national_mean", "national_std"])

    # Ensure numeric dtypes for ML (avoid object columns from reindex/fill)
    numeric_cols = [
        "event_count", "fraud_count", "amount_sum",
        "lag_1d", "lag_7d", "lag_14d", "lag_28d",
        "rolling_mean_7d", "rolling_std_7d", "velocity_7d",
        "dow", "month", "is_holiday", "peer_zscore",
    ]
    for col in numeric_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0)

    return daily.sort_values(["region_id", "date"]).reset_index(drop=True)


def aggregate_and_save(bronze: pd.DataFrame, tenant_id: str) -> pd.DataFrame:
    """Build gold aggregates and persist to Parquet."""
    gold = build_region_daily(bronze, tenant_id)
    if not gold.empty:
        gold.to_parquet(gold_path(tenant_id), index=False)
    return gold


def load_gold(tenant_id: str) -> pd.DataFrame:
    """Load gold aggregates from disk."""
    path = gold_path(tenant_id)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df
