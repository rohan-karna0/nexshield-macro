"""
Ingestion orchestration — adapters → bronze.

Adapters produce List[FraudSignalEvent]; this module writes them to Parquet.
"""

from __future__ import annotations

from typing import Callable

from nexshield.schemas import FraudSignalEvent
from services.ingestion.store import append_bronze, replace_bronze


def ingest_from_adapter(
    adapter_fn: Callable[..., list[FraudSignalEvent]],
    tenant_id: str,
    *,
    replace: bool = False,
    **adapter_kwargs,
) -> dict:
    """
    Run an adapter function and persist results to bronze.

    Parameters
    ----------
    adapter_fn : callable
        e.g. adapters.synthetic_campaign.generate_synthetic_events
    tenant_id : str
        Tenant namespace for partitioned storage.
    replace : bool
        If True, overwrite bronze. If False, append (dedupe by event_id).

    Returns
    -------
    dict with keys: tenant_id, events_written, mode
    """
    events = adapter_fn(tenant_id=tenant_id, **adapter_kwargs)
    if replace:
        count = replace_bronze(events, tenant_id)
        mode = "replace"
    else:
        count = append_bronze(events, tenant_id)
        mode = "append"

    return {
        "tenant_id": tenant_id,
        "events_written": count,
        "events_ingested": len(events),
        "mode": mode,
    }
