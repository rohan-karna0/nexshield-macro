"""
Central path and default settings for the macro pipeline.

All services read/write under DATA_DIR so you can delete `data/` and re-run
the pipeline from scratch at any time.
"""

from pathlib import Path

# Repository root (parent of the `nexshield` package)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Where parquet files and model artifacts live
DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
GOLD_DIR = DATA_DIR / "gold" / "region_daily"
BENCHMARK_DIR = ROOT_DIR / "ml" / "benchmarks"
MODEL_DIR = DATA_DIR / "models"

# Default tenant for local demos
DEFAULT_TENANT = "tenant_demo"

# Forecast settings used by training modules
FORECAST_HORIZON_DAYS = 7
BACKTEST_TRAIN_DAYS = 90
BACKTEST_STEP_DAYS = 7
