"""
Rolling-origin backtest framework for macro forecasting.

Why rolling backtest?
---------------------
Fraud time series must NEVER use a random train/test split — that leaks
future information. Instead we simulate production: at each cutoff date T,
train only on past data and score predictions on the next H days.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nexshield.config import BACKTEST_STEP_DAYS, BACKTEST_TRAIN_DAYS, FORECAST_HORIZON_DAYS


@dataclass
class BacktestFold:
    """One rolling window: train ending at cutoff, test on the next horizon days."""

    cutoff: pd.Timestamp
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame


@dataclass
class BacktestMetrics:
    """Aggregated scores across all folds and regions."""

    model_name: str
    mape_h1: float
    mape_h7: float
    rmse_h1: float
    rmse_h7: float
    n_folds: int
    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "mape_h1": round(self.mape_h1, 4),
            "mape_h7": round(self.mape_h7, 4),
            "rmse_h1": round(self.rmse_h1, 4),
            "rmse_h7": round(self.rmse_h7, 4),
            "n_folds": self.n_folds,
        }


def iter_rolling_folds(
    gold: pd.DataFrame,
    *,
    train_days: int = BACKTEST_TRAIN_DAYS,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    step_days: int = BACKTEST_STEP_DAYS,
) -> list[BacktestFold]:
    """
    Yield rolling train/test splits from gold data.

    Timeline (example):
        |---- train_days ----|-- horizon --|
                              ^ cutoff
    Then step forward by step_days and repeat.
    """
    if gold.empty:
        return []

    dates = sorted(gold["date"].unique())
    min_date = pd.Timestamp(dates[0])
    max_date = pd.Timestamp(dates[-1])

    folds: list[BacktestFold] = []
    cutoff = min_date + pd.Timedelta(days=train_days)

    while cutoff + pd.Timedelta(days=horizon_days) <= max_date:
        train_start = cutoff - pd.Timedelta(days=train_days)
        test_end = cutoff + pd.Timedelta(days=horizon_days)

        train_mask = (gold["date"] > train_start) & (gold["date"] <= cutoff)
        test_mask = (gold["date"] > cutoff) & (gold["date"] <= test_end)

        train_rows = gold.loc[train_mask].copy()
        test_rows = gold.loc[test_mask].copy()

        if not train_rows.empty and not test_rows.empty:
            folds.append(BacktestFold(cutoff=cutoff, train_rows=train_rows, test_rows=test_rows))

        cutoff += pd.Timedelta(days=step_days)

    return folds


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — undefined when y_true=0, so we mask zeros."""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_predictions(
    model_name: str,
    predictions: pd.DataFrame,
    *,
    horizon_col: str = "horizon",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> BacktestMetrics:
    """
    Score a predictions DataFrame with columns:
    horizon, y_true, y_pred (and optionally region_id, cutoff).
    """
    h1 = predictions[predictions[horizon_col] == 1]
    h7 = predictions[predictions[horizon_col] == 7]

    mape_h1 = _safe_mape(h1[y_true_col].values, h1[y_pred_col].values)
    mape_h7 = _safe_mape(h7[y_true_col].values, h7[y_pred_col].values)
    rmse_h1 = _rmse(h1[y_true_col].values, h1[y_pred_col].values)
    rmse_h7 = _rmse(h7[y_true_col].values, h7[y_pred_col].values)

    n_folds = predictions["cutoff"].nunique() if "cutoff" in predictions.columns else 1

    return BacktestMetrics(
        model_name=model_name,
        mape_h1=mape_h1,
        mape_h7=mape_h7,
        rmse_h1=rmse_h1,
        rmse_h7=rmse_h7,
        n_folds=n_folds,
    )


def naive_forecast_fold(fold: BacktestFold, horizon_days: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """
    Baseline: predict each region's next days = last observed event_count.

    This is the model every serious pipeline must beat.
    """
    rows = []
    last_train = fold.train_rows.sort_values("date").groupby("region_id").tail(1)

    for _, state in last_train.iterrows():
        region = state["region_id"]
        last_value = float(state["event_count"])
        test_region = fold.test_rows[fold.test_rows["region_id"] == region].sort_values("date")

        for h, (_, test_row) in enumerate(test_region.iterrows(), start=1):
            if h > horizon_days:
                break
            rows.append(
                {
                    "cutoff": fold.cutoff,
                    "region_id": region,
                    "date": test_row["date"],
                    "horizon": h,
                    "y_true": float(test_row["event_count"]),
                    "y_pred": last_value,
                }
            )

    return pd.DataFrame(rows)


def run_naive_backtest(gold: pd.DataFrame) -> BacktestMetrics:
    """Run naive baseline across all rolling folds."""
    all_preds = []
    for fold in iter_rolling_folds(gold):
        all_preds.append(naive_forecast_fold(fold))
    if not all_preds:
        return BacktestMetrics("naive", np.nan, np.nan, np.nan, np.nan, 0)
    return evaluate_predictions("naive", pd.concat(all_preds, ignore_index=True))
