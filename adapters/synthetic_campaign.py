"""
Synthetic fraud campaign generator.

Produces realistic bronze-layer events so you can run the full pipeline
without downloading external datasets. Injects 3 "campaign" spikes so
backtests have something interesting to predict.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from random import Random

from nexshield.config import DEFAULT_TENANT
from nexshield.schemas import FraudSignalEvent, SignalType

# Demo regions — any real deployment would load these from tenant config
REGIONS = [
    ("US-NY-NYC", 40.7128, -74.0060),
    ("US-CA-LA", 34.0522, -118.2437),
    ("US-TX-HOU", 29.7604, -95.3698),
    ("US-FL-MIA", 25.7617, -80.1918),
    ("US-IL-CHI", 41.8781, -87.6298),
    ("US-WA-SEA", 47.6062, -122.3321),
    ("US-GA-ATL", 33.7490, -84.3880),
    ("US-MA-BOS", 42.3601, -71.0589),
    ("US-CO-DEN", 39.7392, -104.9903),
    ("US-AZ-PHX", 33.4484, -112.0740),
    ("US-PA-PHL", 39.9526, -75.1652),
    ("US-MI-DET", 42.3314, -83.0458),
    ("US-MN-MSP", 44.9778, -93.2650),
    ("US-OR-PDX", 45.5152, -122.6784),
    ("US-NC-CLT", 35.2271, -80.8431),
]


def generate_synthetic_events(
    *,
    tenant_id: str = DEFAULT_TENANT,
    num_days: int = 120,
    base_events_per_region_per_day: int = 3,
    seed: int = 42,
) -> list[FraudSignalEvent]:
    """
    Build a list of FraudSignalEvent rows spanning `num_days`.

    Normal days: Poisson-like counts around `base_events_per_region_per_day`.
    Campaign days: multiply volume by 8x in targeted regions for 5-day windows.
    """
    rng = Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events: list[FraudSignalEvent] = []

    # Three injected campaigns at different times/regions (ground truth for demos)
    campaigns = [
        {"region": "US-NY-NYC", "start_day": 40, "duration": 5, "multiplier": 10},
        {"region": "US-TX-HOU", "start_day": 70, "duration": 5, "multiplier": 8},
        {"region": "US-FL-MIA", "start_day": 95, "duration": 6, "multiplier": 9},
    ]

    def campaign_multiplier(region_id: str, day_index: int) -> float:
        """Return volume multiplier if (region, day) is inside a campaign window."""
        for c in campaigns:
            if c["region"] == region_id and c["start_day"] <= day_index < c["start_day"] + c["duration"]:
                return float(c["multiplier"])
        return 1.0

    for day_index in range(num_days):
        day_start = start + timedelta(days=day_index)

        for region_id, lat, lon in REGIONS:
            mult = campaign_multiplier(region_id, day_index)
            # Jitter daily count — fraud is noisy
            daily_count = max(0, int(rng.gauss(base_events_per_region_per_day * mult, 1.5)))

            for _ in range(daily_count):
                # Spread events across the day
                hour_offset = rng.randint(0, 23)
                minute_offset = rng.randint(0, 59)
                ts = day_start + timedelta(hours=hour_offset, minutes=minute_offset)

                events.append(
                    FraudSignalEvent(
                        tenant_id=tenant_id,
                        timestamp=ts,
                        region_id=region_id,
                        lat=lat + rng.uniform(-0.05, 0.05),
                        lon=lon + rng.uniform(-0.05, 0.05),
                        signal_type=SignalType.TRANSACTION_FRAUD,
                        amount_usd=round(rng.uniform(50, 5000), 2),
                        fraud_label=True,
                        source="synthetic",
                        metadata={"campaign_boost": mult > 1.0},
                    )
                )

    return events
