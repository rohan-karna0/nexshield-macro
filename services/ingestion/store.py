"""
Bronze-layer persistence (Parquet).

In production this would be S3 + Iceberg/Delta; for the portfolio we use
local Parquet partitioned logically by tenant_id.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nexshield.config import BRONZE_DIR
from nexshield.schemas import FraudSignalEvent


def bronze_path(tenant_id: str) -> Path:
    """Path to the bronze parquet file for one tenant."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    return BRONZE_DIR / f"{tenant_id}_events.parquet"


def events_to_dataframe(events: list[FraudSignalEvent]) -> pd.DataFrame:
    """Convert Pydantic models to a flat DataFrame for Parquet."""
    if not events:
        return pd.DataFrame()
    rows = [e.model_dump(mode="json") for e in events]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_bronze(tenant_id: str) -> pd.DataFrame:
    """Read all bronze events for a tenant (empty DataFrame if missing)."""
    path = bronze_path(tenant_id)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def append_bronze(events: list[FraudSignalEvent], tenant_id: str) -> int:
    """
    Append new events to bronze storage.

    Returns the total row count after append.
    """
    incoming = events_to_dataframe(events)
    if incoming.empty:
        return len(load_bronze(tenant_id))

    existing = load_bronze(tenant_id)
    combined = pd.concat([existing, incoming], ignore_index=True)

    # Idempotent ingest: drop duplicate event_ids
    if "event_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_parquet(bronze_path(tenant_id), index=False)
    return len(combined)


def replace_bronze(events: list[FraudSignalEvent], tenant_id: str) -> int:
    """Overwrite bronze with a fresh dataset (used by the demo pipeline)."""
    df = events_to_dataframe(events)
    path = bronze_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)
