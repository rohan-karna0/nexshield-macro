"""
FastAPI ingestion API.

Start with:
    uvicorn services.ingestion.api:app --reload --port 8000

Then POST JSON events to /v1/events:batch
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from nexshield.schemas import BatchIngestRequest, FraudSignalEvent
from services.ingestion.store import append_bronze, load_bronze

app = FastAPI(
    title="NexShield Ingestion",
    description="Accept fraud_signal_events and write to bronze Parquet.",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Liveness probe for Docker / load balancers."""
    return {"status": "ok", "service": "ingestion"}


@app.post("/v1/events:batch")
def ingest_batch(body: BatchIngestRequest):
    """
    Batch ingest fraud signals.

    All events in one request should share the same tenant_id (first event wins).
    """
    if not body.events:
        raise HTTPException(status_code=400, detail="events list is empty")

    tenant_id = body.events[0].tenant_id
    total = append_bronze(body.events, tenant_id)
    return {
        "tenant_id": tenant_id,
        "accepted": len(body.events),
        "bronze_total_rows": total,
    }


@app.get("/v1/tenants/{tenant_id}/events/count")
def event_count(tenant_id: str):
    """Debug endpoint — how many bronze rows exist for a tenant."""
    df = load_bronze(tenant_id)
    return {"tenant_id": tenant_id, "count": len(df)}
