# %%
from pathlib import Path

import mlflow
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
    "model_strategy": {
        "type": "global",  # "by_family_group",  # Options: "global", "by_family", "by_family_group"
        "family_groups": {
            "grocery": ["GROCERY I", "GROCERY II"],
            "beverages": ["BEVERAGES", "LIQUOR,WINE,BEER"],
            "perishable": ["PRODUCE", "DELI", "MEATS", "SEAFOOD"],
            "household": ["CLEANING", "HOME CARE", "PERSONAL CARE"],
            # Add more logical groupings
        },
    },
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
    "evaluation": {
        "mase_seasonality": 7,
        "backtest": {"enabled": True, "folds": 4, "fold_horizon": 14, "step": 14},
    },
}


def train_models_by_group(df_feat, cfg):
    """Train separate models for each family group"""

    strategy = cfg["model_strategy"]["type"]
    results = {}

    if strategy == "global":
        # Single model for all families (current approach)
        results["global"] = train_single_model(df_feat, cfg, "global")

    elif strategy == "by_family":
        # One model per family
        families = (
            df_feat.select("family").distinct().rdd.flatMap(lambda x: x).collect()
        )
        for family in families:
            df_family = df_feat.filter(F.col("family") == family)
            results[family] = train_single_model(df_family, cfg, family)

    elif strategy == "by_family_group":
        # One model per family group
        for group_name, families in cfg["model_strategy"]["family_groups"].items():
            df_group = df_feat.filter(F.col("family").isin(families))
            results[group_name] = train_single_model(df_group, cfg, group_name)

    return results


def train_single_model(df_feat, cfg, model_name):
    """Train and evaluate a single model"""

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", cfg["model"]["type"])

        # Split
        train, test = train_test_split(
            df_feat,
            cfg["data"]["date_col"],
            cfg["data"]["group_cols"],
            cfg["split"]["mode"],
            cfg["split"]["train_end_date"],
            cfg["split"]["test_horizon"],
            cfg["data"]["min_train_periods"],
        )

        # Get feature columns
        feature_cols = [
            c
            for c in train.columns
            if c
            not in cfg["data"]["group_cols"]
            + [cfg["data"]["date_col"], cfg["data"]["target_col"], "label"]
        ]

        # Train
        est = model_factory(cfg["model"]["type"], cfg["model"]["params"])
        ic(est)
        model = fit_global_model(
            train.sample(0.001),  # Use full training data!
            cfg["data"]["target_col"],
            cfg["data"]["group_cols"],
            feature_cols,
            est,
        )
        ic()

        # Predict
        pred = predict_global(
            model,
            test,
            cfg["data"]["group_cols"],
            cfg["data"]["date_col"],
            cfg["data"]["target_col"],
        )
        ic()

        # Evaluate
        by_series, portfolio = compute_metrics(
            pred,
            cfg["data"]["date_col"],
            "y",
            "prediction",
            cfg["data"]["group_cols"],
            cfg["evaluation"]["mase_seasonality"],
        )
        ic()

        # Log metrics
        portfolio_metrics = portfolio.first().asDict()
        ic(portfolio_metrics)
        for key, value in portfolio_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_param(key, str(value))

        # Log model
        if spark.sparkContext.master.startswith("spark"):
            mlflow.spark.log_model(model, f"model_{model_name}")
        else:
            print("Skipping Spark model logging (not running on Databricks)")

        return {
            "model": model,
            "predictions": pred,
            "metrics": {"by_series": by_series, "portfolio": portfolio},
        }


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/daniel.more.torres@gmail.com/favorita_gbt_regressor")

with mlflow.start_run(run_name="favorita_multi_model"):
    mlflow.log_param("strategy", cfg["model_strategy"]["type"])
    mlflow.log_param("cfg", cfg)

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

    # --- Step 3: Train model based on strategy ---
    ic("# --- Step 3: Train model based on strategy ---")
    all_results = train_models_by_group(df_feat, cfg)

    # Compare results across groups
    comparison = {}
    for model_name, result in all_results.items():
        portfolio_metrics = result["metrics"]["portfolio"].first().asDict()
        comparison[model_name] = portfolio_metrics
        ic(f"{model_name}: {portfolio_metrics}")

    # Log comparison
    import pandas as pd

    comparison_df = pd.DataFrame(comparison).T
    mlflow.log_table(comparison_df, "model_comparison.json")
