"""
Shared data contracts for ingestion and macro ETL.

Every adapter (CSV, synthetic, webhook) must produce rows matching
FraudSignalEvent so downstream ETL stays source-agnostic.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Kind of fraud signal — extend as you add adapters."""

    TRANSACTION_FRAUD = "transaction_fraud"
    COMPLAINT = "complaint"
    CHARGEBACK = "chargeback"
    REPORT = "report"


class FraudSignalEvent(BaseModel):
    """
    Bronze-layer event: one reported or detected fraud signal.

    This is the atomic input to the macro wave predictor. We count these
    per (tenant, region, day) to build the gold time series.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    timestamp: datetime
    region_id: str
    lat: float | None = None
    lon: float | None = None
    signal_type: SignalType = SignalType.TRANSACTION_FRAUD
    amount_usd: float | None = None
    fraud_label: bool | None = True
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    """HTTP body for POST /v1/events:batch."""

    events: list[FraudSignalEvent]
