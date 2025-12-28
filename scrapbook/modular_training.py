# %%
from pathlib import Path

import yaml
from icecream import ic

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


config["databricks"]["catalog"], config["databricks"]["schema"], config["databricks"][
    "volume"
]


# %%
def running_on_databricks():
    """Detect if running in Databricks environment"""
    try:
        import pyspark.dbutils  # only available in Databricks

        return True
    except ImportError:
        return False


IS_DATABRICKS = running_on_databricks()
ic(IS_DATABRICKS)

from helper import (
    aggregate_to_granularity,
    assemble_global_pipeline,
    assert_unique_series_rows,
    build_features,
    compute_metrics,
    fit_global_model,
    fit_predict_local,
    model_factory,
    plot_forecast,
    plot_train_test_forecast,
    predict_global,
    rolling_backtest,
    run_forecast,
    train_test_split,
)

# %%
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# from helper import run_forecast, aggregate_to_granularity, build_features, train_test_split


# Give Spark way more memory since you have 32GB RAM available
spark = (
    SparkSession.builder.appName("TimeSeriesForecast")
    .config("spark.driver.memory", "12g")
    .config("spark.executor.memory", "12g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.default.parallelism", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)


# %%
# Load data (must include columns: date, sales, family, store_nbr)
if IS_DATABRICKS:
    df_raw = (
        spark.read.format("delta")
        .table("portfolio_catalog.databricks_pipeline.silver_training")
        .withColumn("date", F.to_date(F.col("date")))
    )

else:
    df_raw = spark.read.parquet("notebooks/data/train.parquet").withColumn(
        "date", F.to_date(F.col("date"))
    )

ic(IS_DATABRICKS)

# %%
cfg = {
    "data": {
        "date_col": "date",
        "target_col": "sales",
        "group_cols": ["family", "store_nbr"],
        "freq": "D",
        "min_train_periods": 56,
    },
    "aggregation": {
        "target_agg": "sum",
        "extra_numeric_aggs": {"dcoilwtico": "mean", "onpromotion": "sum"},
    },
    "features": {"lags": [1, 7, 14, 28], "mas": [7, 28], "add_time_signals": True},
    "split": {"mode": "horizon", "train_end_date": "", "test_horizon": 28},
    "model": {"type": "spark_gbt", "params": {"maxDepth": 7, "maxIter": 120}},
    # "model": {"type": "spark_lgbt", "params": {"maxDepth": 7, "maxIter": 120}},
    "evaluation": {
        "mase_seasonality": 7,
        "backtest": {"enabled": True, "folds": 4, "fold_horizon": 14, "step": 14},
    },
}

# --- Step 1: Features ---
ic("# --- Step 1: Features ---")
df_feat = build_features(
    df_raw,
    cfg["data"]["date_col"],
    cfg["data"]["target_col"],
    cfg["data"]["group_cols"],
    cfg["features"]["lags"],
    cfg["features"]["mas"],
    cfg["features"]["add_time_signals"],
    pre_aggregate=True,
    target_agg=cfg["aggregation"]["target_agg"],
    extra_numeric_aggs=cfg["aggregation"].get("extra_numeric_aggs"),
)
ic(df_feat.limit(5))

# --- Step 2: Split ---
ic("# --- Step 2: Split ---")
train, test = train_test_split(
    df_feat,
    cfg["data"]["date_col"],
    cfg["data"]["group_cols"],
    cfg["split"]["mode"],
    cfg["split"]["train_end_date"],
    cfg["split"]["test_horizon"],
    cfg["data"]["min_train_periods"],
)

# --- Step 3: Train (global model) ---
ic("# --- Step 3: Train (global model) ---")
est = model_factory(cfg["model"]["type"], cfg["model"]["params"])
feature_cols = [
    c
    for c in train.columns
    if c
    not in cfg["data"]["group_cols"]
    + [cfg["data"]["date_col"], cfg["data"]["target_col"], "label"]
]
ic(feature_cols)
model = fit_global_model(
    train.sample(0.01),
    cfg["data"]["target_col"],
    cfg["data"]["group_cols"],
    feature_cols,
    est,
)

# --- Step 4: Predict ---
ic("# --- Step 4: Predict ---")
pred = predict_global(
    model,
    test,
    cfg["data"]["group_cols"],
    cfg["data"]["date_col"],
    cfg["data"]["target_col"],
)
ic(pred.show(10))

# --- Step 5: Metrics ---
by_series, portfolio = compute_metrics(
    pred,
    cfg["data"]["date_col"],
    "y",
    "prediction",
    cfg["data"]["group_cols"],
    cfg["evaluation"]["mase_seasonality"],
)
print(by_series.orderBy("wMAPE").show(150))
print(portfolio.show())
