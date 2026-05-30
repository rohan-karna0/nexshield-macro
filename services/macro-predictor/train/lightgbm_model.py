"""
Macro forecaster — LightGBM champion with sklearn fallback.

LightGBM is preferred in production (fast, great on tabular lags). If the
native DLL is unavailable (e.g. Python 3.14), we fall back to scikit-learn's
HistGradientBoostingRegressor so the pipeline always runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from nexshield.config import FORECAST_HORIZON_DAYS, MODEL_DIR
from train.backtest import (
    BacktestMetrics,
    evaluate_predictions,
    iter_rolling_folds,
)

try:
    import lightgbm as lgb

    # Smoke-test native library (can import but fail at runtime on some Python versions)
    _probe = lgb.LGBMRegressor(n_estimators=1)
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError, FileNotFoundError, ValueError):
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore


FEATURE_COLUMNS = [
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "rolling_mean_7d",
    "rolling_std_7d",
    "velocity_7d",
    "dow",
    "month",
    "is_holiday",
    "peer_zscore",
    "amount_sum",
]

TARGET_COLUMN = "event_count"
MODEL_BACKEND = "lightgbm" if LIGHTGBM_AVAILABLE else "sklearn_hist_gb"


class RegressorModel(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    @property
    def feature_importances_(self) -> np.ndarray: ...


def _prepare_xy(df: pd.DataFrame, region_encoder: LabelEncoder) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    work["region_code"] = region_encoder.transform(work["region_id"])
    feature_cols = FEATURE_COLUMNS + ["region_code"]
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)
    y = work[TARGET_COLUMN].values.astype(float)
    return X, y


def _build_model() -> RegressorModel:
    if LIGHTGBM_AVAILABLE:
        return lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            n_estimators=200,
        )
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=200,
        random_state=42,
    )


def train_lightgbm(
    gold: pd.DataFrame,
    *,
    save_path: Path | None = None,
) -> tuple[RegressorModel, LabelEncoder, list[str], str]:
    """
    Fit one global model on all historical gold rows.

    Returns (model, region_encoder, feature_names, backend_name).
    """
    gold = gold.dropna(subset=["lag_28d"]).copy()
    if gold.empty:
        raise ValueError("Not enough gold data to train — run ETL first.")

    encoder = LabelEncoder()
    encoder.fit(gold["region_id"])
    X, y = _prepare_xy(gold, encoder)
    feature_names = list(X.columns)
    y_log = np.log1p(y)

    model = _build_model()
    model.fit(X, y_log)

    if save_path is None:
        save_path = MODEL_DIR / f"macro_{MODEL_BACKEND}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if LIGHTGBM_AVAILABLE and hasattr(model, "booster_"):
        model.booster_.save_model(str(save_path.with_suffix(".txt")))

    return model, encoder, feature_names, MODEL_BACKEND


def predict_next_day(
    model: RegressorModel,
    region_encoder: LabelEncoder,
    row: pd.Series,
) -> float:
    frame = row.to_frame().T
    frame["region_code"] = region_encoder.transform(frame["region_id"])
    X = frame[FEATURE_COLUMNS + ["region_code"]].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float64)
    pred_log = model.predict(X)
    return float(np.expm1(pred_log[0]))


def _recursive_horizon_forecast(
    model: RegressorModel,
    encoder: LabelEncoder,
    history: pd.DataFrame,
    region_id: str,
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    region_hist = history[history["region_id"] == region_id].sort_values("date").copy()
    working = region_hist[region_hist["date"] <= cutoff].copy()
    future_dates = pd.date_range(
        cutoff + pd.Timedelta(days=1),
        cutoff + pd.Timedelta(days=horizon_days),
        freq="D",
        tz="UTC",
    )

    rows = []
    for h, future_date in enumerate(future_dates, start=1):
        if working.empty:
            break
        last = working.iloc[-1].copy()
        last["date"] = future_date
        last["dow"] = future_date.dayofweek
        last["month"] = future_date.month

        y_pred = max(0.0, predict_next_day(model, encoder, last))
        new_row = last.copy()
        new_row["event_count"] = y_pred
        new_row["lag_1d"] = working.iloc[-1]["event_count"]
        working = pd.concat([working, new_row.to_frame().T], ignore_index=True)

        rows.append(
            {
                "cutoff": cutoff,
                "region_id": region_id,
                "date": future_date,
                "horizon": h,
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)


def backtest_lightgbm(gold: pd.DataFrame) -> tuple[BacktestMetrics, pd.DataFrame]:
    all_predictions = []

    for fold in iter_rolling_folds(gold):
        train = fold.train_rows.dropna(subset=["lag_28d"])
        if train.empty:
            continue

        model, encoder, _, backend = train_lightgbm(train)
        model_label = f"macro_{backend}"

        for region_id in train["region_id"].unique():
            preds = _recursive_horizon_forecast(
                model, encoder, gold, region_id, fold.cutoff, FORECAST_HORIZON_DAYS
            )
            if preds.empty:
                continue

            test_region = fold.test_rows[fold.test_rows["region_id"] == region_id][
                ["date", "event_count"]
            ]
            merged = preds.merge(test_region, on="date", how="inner")
            merged = merged.rename(columns={"event_count": "y_true"})
            all_predictions.append(merged)

    if not all_predictions:
        empty = BacktestMetrics("macro_model", np.nan, np.nan, np.nan, np.nan, 0)
        return empty, pd.DataFrame()

    pred_df = pd.concat(all_predictions, ignore_index=True)
    metrics = evaluate_predictions(f"macro_{MODEL_BACKEND}", pred_df)
    return metrics, pred_df


def feature_importance_report(model: RegressorModel, feature_names: list[str], top_k: int = 8) -> pd.DataFrame:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame({"feature": feature_names[:top_k], "importance": [0] * min(top_k, len(feature_names))})
    report = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return report
