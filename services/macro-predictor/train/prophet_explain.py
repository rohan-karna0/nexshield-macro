"""
Prophet explainability branch — interpretable seasonality for top regions.

LightGBM wins on accuracy; Prophet wins on explainability ("Fridays are
higher risk because of weekly seasonality"). We fit Prophet only on the
top-N busiest regions to keep training time reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nexshield.config import FORECAST_HORIZON_DAYS
from train.backtest import (
    BacktestMetrics,
    evaluate_predictions,
    iter_rolling_folds,
)

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


@dataclass
class ProphetExplanation:
    """Decomposed forecast components for one region."""

    region_id: str
    trend: float
    weekly: float
    yearly: float
    forecast_next_7d: list[float]


def _to_prophet_df(region_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet expects columns ds (datetime) and y (numeric target).

    We use event_count as y — daily fraud signal volume for the region.
    """
    pdf = region_daily[["date", "event_count"]].copy()
    pdf = pdf.rename(columns={"date": "ds", "event_count": "y"})
    pdf["ds"] = pdf["ds"].dt.tz_localize(None)
    return pdf


def fit_prophet_region(region_daily: pd.DataFrame) -> Prophet:
    """Train one Prophet model on a single region's history."""
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet is not installed. Run: pip install prophet")

    pdf = _to_prophet_df(region_daily)
    if len(pdf) < 14:
        raise ValueError("Need at least 14 days of data for Prophet.")

    # daily_seasonality=False because we only have one point per day
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(pdf)
    return model


def explain_region(model: Prophet, region_id: str, history_days: int = 30) -> ProphetExplanation:
    """
    Extract trend/seasonality from the last forecast and return next-7-day path.

    Useful for dashboard tooltips: "Risk elevated due to +2.1 weekly effect."
    """
    future = model.make_future_dataframe(periods=FORECAST_HORIZON_DAYS)
    forecast = model.predict(future)

    tail = forecast.tail(FORECAST_HORIZON_DAYS)
    last_components = forecast.iloc[-FORECAST_HORIZON_DAYS - 1]

    return ProphetExplanation(
        region_id=region_id,
        trend=float(last_components.get("trend", 0)),
        weekly=float(last_components.get("weekly", 0)),
        yearly=float(last_components.get("yearly", 0)),
        forecast_next_7d=[max(0.0, float(v)) for v in tail["yhat"].tolist()],
    )


def backtest_prophet_top_regions(
    gold: pd.DataFrame,
    top_n: int = 5,
) -> tuple[BacktestMetrics, pd.DataFrame, list[ProphetExplanation]]:
    """
    Rolling backtest Prophet on the top-N regions by total event volume.

    Returns metrics, prediction DataFrame, and explanation objects for the
    final full-data fit (for demo printing).
    """
    if not PROPHET_AVAILABLE:
        empty = BacktestMetrics("prophet", float("nan"), float("nan"), float("nan"), float("nan"), 0)
        return empty, pd.DataFrame(), []

    volume = gold.groupby("region_id")["event_count"].sum().sort_values(ascending=False)
    top_regions = volume.head(top_n).index.tolist()

    all_predictions = []

    for fold in iter_rolling_folds(gold):
        for region_id in top_regions:
            train_region = fold.train_rows[fold.train_rows["region_id"] == region_id]
            test_region = fold.test_rows[fold.test_rows["region_id"] == region_id]

            if len(train_region) < 21 or test_region.empty:
                continue

            try:
                model = fit_prophet_region(train_region)
            except Exception:
                continue

            future = model.make_future_dataframe(periods=FORECAST_HORIZON_DAYS)
            forecast = model.predict(future)
            forecast_tail = forecast.tail(FORECAST_HORIZON_DAYS)[["ds", "yhat"]]

            test_sorted = test_region.sort_values("date").head(FORECAST_HORIZON_DAYS)
            for h, (_, test_row) in enumerate(test_sorted.iterrows(), start=1):
                if h - 1 >= len(forecast_tail):
                    break
                y_pred = max(0.0, float(forecast_tail.iloc[h - 1]["yhat"]))
                all_predictions.append(
                    {
                        "cutoff": fold.cutoff,
                        "region_id": region_id,
                        "date": test_row["date"],
                        "horizon": h,
                        "y_true": float(test_row["event_count"]),
                        "y_pred": y_pred,
                    }
                )

    if not all_predictions:
        empty = BacktestMetrics("prophet", float("nan"), float("nan"), float("nan"), float("nan"), 0)
        return empty, pd.DataFrame(), []

    pred_df = pd.concat([pd.DataFrame([r]) for r in all_predictions], ignore_index=True)
    metrics = evaluate_predictions("prophet", pred_df)

    # Final explanations on full history (demo / dashboard)
    explanations: list[ProphetExplanation] = []
    for region_id in top_regions:
        region_all = gold[gold["region_id"] == region_id]
        if len(region_all) < 21:
            continue
        try:
            model = fit_prophet_region(region_all)
            explanations.append(explain_region(model, region_id))
        except Exception:
            continue

    return metrics, pred_df, explanations
