!pip install -q mlflow lightgbm
!pip install icecream
%restart_python
# %%
import os
from pathlib import Path

import mlflow
import yaml
import lightgbm as lgb
import pandas as pd
import numpy as np
from icecream import ic
from pyspark.sql.types import StringType


with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

config["databricks"]["catalog"], config["databricks"]["schema"], config["databricks"]["volume"]

# %%
def running_on_databricks():
    """Detect if running in Databricks environment"""
    try:
        import pyspark.dbutils
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

spark = SparkSession.builder.getOrCreate()

# %%
# Load data
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
        "type": "global",  # Options: "global", "by_family", "by_family_group"
        "family_groups": {
            "cluster_low_errors": {"BEVERAGES", "GROCERY I", "PRODUCE", "CLEANING"},
            "cluster_high_errors": [
                "SCHOOL AND OFFICE SUPPLIES",
                "SEAFOOD",
                "BOOKS",
                "LAWN AND GARDEN",
                "EGGS",
            ],
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
    "split": {
        "mode": "horizon", 
        "train_end_date": "", 
        "test_horizon": 28,
        "validation_size": 0.15  # Added for validation split
    },
    "model": {
        "type": "lgbm",  # Changed to lgbm
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1
        },
        "early_stopping_rounds": 30
    },
    "evaluation": {
        "mase_seasonality": 7,
        "backtest": {"enabled": True, "folds": 4, "fold_horizon": 14, "step": 14},
    },
}


def train_lgbm_model(train_pdf, val_pdf, feature_cols, target_col, params, early_stopping_rounds=30):
    """Train LightGBM model with Pandas DataFrames"""
    
    # Prepare training data
    X_train = train_pdf[feature_cols]
    y_train = train_pdf[target_col]
    
    # Apply log transformation (optional, adjust as needed)
    y_train_log = np.log1p(y_train)
    
    # Prepare validation data
    X_val = val_pdf[feature_cols]
    y_val = val_pdf[target_col]
    y_val_log = np.log1p(y_val)
    
    # Train model
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )
    
    return model


def predict_lgbm_model(model, test_pdf, feature_cols):
    """Generate predictions with LightGBM model"""
    X_test = test_pdf[feature_cols]
    predictions_log = model.predict(X_test)
    
    # Reverse log transformation
    predictions = np.expm1(predictions_log)
    
    return predictions


def train_single_model(df_feat, cfg, model_name):
    """Train and evaluate a single LightGBM model"""
    
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", cfg["model"]["type"])
        
        # Split into train/test
        train_spark, test_spark = train_test_split(
            df_feat,
            cfg["data"]["date_col"],
            cfg["data"]["group_cols"],
            cfg["split"]["mode"],
            cfg["split"]["train_end_date"],
            cfg["split"]["test_horizon"],
            cfg["data"]["min_train_periods"],
        )
        
        # Convert to Pandas
        ic("Converting to Pandas...")
        train_pdf = train_spark.toPandas()
        test_pdf = test_spark.toPandas()
        
        # Further split train into train/validation
        train_pdf = train_pdf.sort_values(cfg["data"]["date_col"])
        val_size = int(len(train_pdf) * cfg["split"]["validation_size"])
        
        val_pdf = train_pdf.tail(val_size).copy()
        train_pdf = train_pdf.head(len(train_pdf) - val_size).copy()
        
        ic(f"Train size: {len(train_pdf)}, Val size: {len(val_pdf)}, Test size: {len(test_pdf)}")
        
        # Get feature columns
        feature_cols = [
            c for c in train_pdf.columns
            if c not in cfg["data"]["group_cols"] + 
            [cfg["data"]["date_col"], cfg["data"]["target_col"], "label"]
        ]
        
        ic(f"Number of features: {len(feature_cols)}")
        
        # Train LightGBM model
        ic("Training LightGBM model...")
        model = train_lgbm_model(
            train_pdf,
            val_pdf,
            feature_cols,
            cfg["data"]["target_col"],
            cfg["model"]["params"],
            cfg["model"]["early_stopping_rounds"]
        )
        
        ic(f"Best iteration: {model.best_iteration_}")
        
        # Generate predictions on test set
        ic("Generating predictions...")
        predictions = predict_lgbm_model(model, test_pdf, feature_cols)
        
        # Add predictions to test dataframe
        test_pdf["prediction"] = predictions
        test_pdf["y"] = test_pdf[cfg["data"]["target_col"]]
        
        # Convert back to Spark for metrics computation
        pred_spark = spark.createDataFrame(test_pdf)
        
        # Evaluate
        ic("Computing metrics...")
        by_series, portfolio = compute_metrics(
            pred_spark,
            cfg["data"]["date_col"],
            "y",
            "prediction",
            cfg["data"]["group_cols"],
            cfg["evaluation"]["mase_seasonality"],
        )
        
        # Log metrics
        agg_df = by_series.groupBy("family").agg(
            F.min("wMAPE").alias("min_wMAPE"),
            F.max("wMAPE").alias("max_wMAPE"),
            F.expr("percentile_approx(wMAPE, 0.5)").alias("median_wMAPE"),
        )
        
        agg_list = agg_df.collect()
        
        for row in agg_list:
            family = row["family"]
            mlflow.log_metric(f"wMAPE_min_{family}", float(row["min_wMAPE"]))
            mlflow.log_metric(f"wMAPE_max_{family}", float(row["max_wMAPE"]))
            mlflow.log_metric(f"wMAPE_median_{family}", float(row["median_wMAPE"]))
        
        portfolio_metrics = portfolio.first().asDict()
        ic(portfolio_metrics)
        
        for key, value in portfolio_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_param(key, str(value))
        
        # Log model with MLflow
        ic("Logging model...")
        input_example = train_pdf[:5]

        mlflow.lightgbm.log_model(model, f"model_{model_name}", input_example=input_example)
        
        # Add metadata columns to predictions
        from pyspark.sql.functions import lit, current_timestamp
        from datetime import datetime
        
        pred_spark = pred_spark.withColumn("scenario", lit("lightgbm")) \
            .withColumn("model_name", lit(model_name)) \
            .withColumn("model_type", lit(cfg["model"]["type"])) \
            .withColumn("model_strategy", lit(cfg["model_strategy"]["type"])) \
            .withColumn("prediction_timestamp", current_timestamp()) \
            .withColumn("model_version", lit(mlflow.active_run().info.run_id)) \
            .withColumn("best_iteration", lit(int(model.best_iteration_)))
        
        # Save predictions with proper partitioning
        if IS_DATABRICKS:
            # Save to Delta table with partitioning for better query performance
            pred_spark.write \
                .format("delta") \
                .mode("append") \
                .partitionBy("model_name", cfg["data"]["date_col"]) \
                .option("mergeSchema", "true") \
                .saveAsTable(f"{config['databricks']['catalog']}.{config['databricks']['schema']}.silver_test_predictions")
            
            ic(f"Saved predictions to Delta table: silver_test_predictions")
        else:
            # Local save with partitioning
            output_path = f"predictions/lgbm_model_{cfg['model_strategy']['type']}_{model_name}"
            pred_spark.write \
                .format("parquet") \
                .mode("overwrite") \
                .partitionBy("model_name") \
                .save(output_path)
            
            ic(f"Saved predictions locally to: {output_path}")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_table(feature_importance.head(20), f"feature_importance_{model_name}.json")
        
        return {
            "model": model,
            "predictions": pred_spark,
            "metrics": {"by_series": by_series, "portfolio": portfolio},
            "feature_importance": feature_importance
        }


def train_models_by_group(df_feat, cfg):
    """Train separate models for each family group"""
    
    strategy = cfg["model_strategy"]["type"]
    results = {}
    
    if strategy == "global":
        # Single model for all families
        results["global"] = train_single_model(df_feat, cfg, model_name="global")
        ic(results["global"])
        
    elif strategy == "by_family":
        # One model per family
        families = df_feat.select("family").distinct().rdd.flatMap(lambda x: x).collect()
        for family in families:
            ic(f"Training model for family: {family}")
            df_family = df_feat.filter(F.col("family") == family)
            results[family] = train_single_model(df_family, cfg, family)
            
    elif strategy == "by_family_group":
        # One model per family group
        for group_name, families in cfg["model_strategy"]["family_groups"].items():
            ic(f"Training model for group: {group_name}")
            df_group = df_feat.filter(F.col("family").isin(families))
            results[group_name] = train_single_model(df_group, cfg, group_name)
    
    return results


# Main execution
   
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/daniel.more.torres@gmail.com/favorita_lgbm_regressor")

with mlflow.start_run(run_name="favorita_lgbm_multi_model"):
    mlflow.log_param("strategy", cfg["model_strategy"]["type"])
    mlflow.log_params(cfg["model"]["params"])
    mlflow.log_param("cfg", str(cfg))
    
    # --- Step 1: Features ---
    ic("# --- Step 1: Build Features ---")
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
    ic(df_feat.schema)
    # # Get list of non-string columns
    # numeric_cols = [field.name for field in df_feat.schema.fields 
    #             if not isinstance(field.dataType, StringType)]

    # # Select only those columns
    # df_feat = df_feat.select(numeric_cols)

    ic(df_feat.limit(5))
    
    # --- Step 2: Train models based on strategy ---
    ic("# --- Step 2: Train Models ---")
    all_results = train_models_by_group(df_feat, cfg)
    
    # Compare results across groups
    comparison = {}
    for model_name, result in all_results.items():
        portfolio_metrics = result["metrics"]["portfolio"].first().asDict()
        comparison[model_name] = portfolio_metrics
        ic(f"{model_name}: {portfolio_metrics}")
    
    # Log comparison
    comparison_df = pd.DataFrame(comparison).T
    mlflow.log_table(comparison_df, "model_comparison.json")
    
    ic("Training complete!")