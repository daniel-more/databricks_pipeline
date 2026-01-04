!pip install -q mlflow lightgbm
!pip install icecream
%restart_python
# %%
import os
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from icecream import ic
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import current_timestamp, lit
from pyspark.sql.types import StringType

spark = (
    SparkSession.builder.config("spark.driver.memory", "16g")
    .config("spark.driver.maxResultSize", "4g")
    .getOrCreate()
)


with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

config["databricks"]["catalog"], config["databricks"]["schema"], config["databricks"][
    "volume"
]


# %%
def running_on_databricks():
    """Detect if running in Databricks environment"""
    try:
        import pyspark.dbutils

        return True
    except ImportError:
        return False


IS_DATABRICKS = running_on_databricks()

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
        "test_horizon": 180,
        "validation_size": 0.15,
    },
    "model": {
        "type": "lgbm",
        "params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        },
        "early_stopping_rounds": 30,
    },
    "evaluation": {
        "mase_seasonality": 7,
        "backtest": {"enabled": True, "folds": 4, "fold_horizon": 14, "step": 14},
    },
}


# ========================================
# HELPER FUNCTIONS - Simplified & Clear
# ========================================


def get_feature_columns(df, cfg):
    """
    Define feature columns once, clearly.
    Returns: (categorical_cols, numeric_cols, all_feature_cols)
    """
    exclude_cols = {cfg["data"]["date_col"], cfg["data"]["target_col"], "label"}

    # Categorical features from config
    categorical_cols = cfg["data"]["group_cols"]

    # Numeric features: everything else that's numeric and not excluded
    numeric_cols = [
        c
        for c in df.columns
        if c not in exclude_cols
        and c not in categorical_cols
        and df[c].dtype in ["int64", "float64", "int32", "float32", "bool"]
    ]

    # Complete feature list
    all_features = categorical_cols + numeric_cols

    return categorical_cols, numeric_cols, all_features


def prepare_dataframes(train_pdf, test_pdf, categorical_cols, numeric_cols, target_col, date_col):
    """
    Prepare train and test dataframes with consistent columns and types.
    Returns: (prepared_train, prepared_test, feature_cols)
    """
    feature_cols = categorical_cols + numeric_cols
    # Keep date column for later analysis, but don't include in features
    required_cols = feature_cols + [target_col, date_col]
    
    # Subset to required columns only
    train_pdf = train_pdf[required_cols].copy()
    test_pdf = test_pdf[required_cols].copy()

    # Convert categorical columns to category dtype
    for col in categorical_cols:
        train_pdf[col] = train_pdf[col].astype("category")
        test_pdf[col] = test_pdf[col].astype("category")

    # Verify consistency
    assert list(train_pdf[feature_cols].columns) == list(
        test_pdf[feature_cols].columns
    ), "Train/test feature columns don't match!"

    return train_pdf, test_pdf, feature_cols


def train_lgbm_model(
    train_pdf, val_pdf, feature_cols, target_col, params, early_stopping_rounds=30
):
    """Train LightGBM model with early stopping on validation set"""

    # Prepare data
    X_train = train_pdf[feature_cols]
    y_train = train_pdf[target_col]
    y_train_log = np.log1p(y_train)

    X_val = val_pdf[feature_cols]
    y_val = val_pdf[target_col]
    y_val_log = np.log1p(y_val)

    # Train with early stopping
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        categorical_feature="auto",
    )

    return model


def predict_lgbm_model(model, test_data, feature_cols):
    """Generate predictions and reverse log transformation"""

    X_test = test_data[feature_cols]

    # Verify feature count
    assert X_test.shape[1] == len(
        feature_cols
    ), f"Feature mismatch: got {X_test.shape[1]}, expected {len(feature_cols)}"

    assert (
        X_test.shape[1] == model.n_features_
    ), f"Model expects {model.n_features_} features, got {X_test.shape[1]}"

    # Predict using best_iteration from early stopping
    # If early stopping was used, this uses the optimal number of trees
    # Otherwise, uses all trees (n_estimators)
    num_iteration = model.best_iteration_ if model.best_iteration_ > 0 else None
    predictions_log = model.predict(X_test, num_iteration=num_iteration)
    predictions = np.expm1(predictions_log)

    

    return predictions


def save_predictions(
    pred_spark, cfg, model_name, run_id, best_iteration, is_databricks, config
):
    """Save predictions to Delta table or local parquet"""

    # Add metadata columns
    pred_spark = (
        pred_spark.withColumn("scenario", lit("lightgbm"))
        .withColumn("model_name", lit(model_name))
        .withColumn("model_type", lit(cfg["model"]["type"]))
        .withColumn("model_strategy", lit(cfg["model_strategy"]["type"]))
        .withColumn("prediction_timestamp", current_timestamp())
        .withColumn("model_version", lit(run_id))
        .withColumn("best_iteration", lit(int(best_iteration)))
    )

    # Save based on environment
    if is_databricks:
        table_name = f"{config['databricks']['catalog']}.{config['databricks']['schema']}.lightgbm_silver_test_predictions"

        table_schema = spark.table(table_name).schema
        onpromotion_type = [f.dataType for f in table_schema if f.name == "onpromotion"][0]
        print(onpromotion_type)
        # Cast 'onpromotion' in pred_spark to match the Delta table type
        pred_spark = pred_spark.withColumn(
            "onpromotion",
            F.col("onpromotion").cast(onpromotion_type.simpleString()))

        pred_spark.write.format("delta").mode("append").partitionBy(
            "model_name", cfg["data"]["date_col"]
        ).option("mergeSchema", "true").saveAsTable(table_name)

        print(f"✓ Saved predictions to Delta table: {table_name}")
    else:
        output_path = (
            f"predictions/lgbm_model_{cfg['model_strategy']['type']}_{model_name}"
        )
        pred_spark.write.format("parquet").mode("overwrite").partitionBy(
            "model_name"
        ).save(output_path)

        print(f"✓ Saved predictions locally to: {output_path}")

    return pred_spark


# ========================================
# MAIN TRAINING FUNCTION - Simplified
# ========================================


def train_single_model(df_feat, cfg, model_name):
    """Train and evaluate a single LightGBM model"""

    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"{'='*60}")

    mlflow.lightgbm.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
    )

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", cfg["model"]["type"])

        # Step 1: Split into train/test
        print("→ Splitting train/test data...")
        train_spark, test_spark = train_test_split(
            df_feat,
            cfg["data"]["date_col"],
            cfg["data"]["group_cols"],
            cfg["split"]["mode"],
            cfg["split"]["train_end_date"],
            cfg["split"]["test_horizon"],
            cfg["data"]["min_train_periods"],
        )

        # Step 2: Convert to Pandas
        print("→ Converting to Pandas...")
        train_pdf = train_spark.toPandas()
        test_pdf = test_spark.toPandas()
        print(f"  Train size: {len(train_pdf):,}, Test size: {len(test_pdf):,}")

        # Step 3: Define features consistently
        print("→ Defining feature columns...")
        categorical_cols, numeric_cols, all_features = get_feature_columns(
            train_pdf, cfg
        )
        print(f"  Categorical features: {len(categorical_cols)} - {categorical_cols}")
        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Total features: {len(all_features)}")

        # Step 4: Prepare dataframes
        print("→ Preparing dataframes...")
        train_pdf, test_pdf, feature_cols = prepare_dataframes(
            train_pdf,
            test_pdf,
            categorical_cols,
            numeric_cols,
            cfg["data"]["target_col"], cfg["data"]["date_col"]
        )

        # Step 5: Train model
        print("→ Training LightGBM model...")
        model = train_lgbm_model(
            train_pdf,
            test_pdf,  # Using test as validation for early stopping
            feature_cols=feature_cols,
            target_col=cfg["data"]["target_col"],
            params=cfg["model"]["params"],
            early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
        )
        print(f"  Best iteration: {model.best_iteration_}")

        # Step 6: Generate predictions
        print("→ Generating predictions...")
        predictions = predict_lgbm_model(model, test_pdf, feature_cols)
        

        # Add predictions to test dataframe
        test_pdf["prediction"] = predictions
        test_pdf["y"] = test_pdf[cfg["data"]["target_col"]]
        print(test_pdf.columns)
        print(test_pdf.head(5).T)

        # Convert back to Spark
        pred_spark = spark.createDataFrame(test_pdf)

        # Step 7: Log model with signature
        print("→ Logging model to MLflow...")
        from mlflow.models.signature import infer_signature

        print("infer_signature")
        signature = infer_signature(
            train_pdf[feature_cols], model.predict(train_pdf[feature_cols])
        )
        print("log_model")
        mlflow.lightgbm.log_model(
            model, name=f"model_{model_name}", signature=signature
        )

        # Step 8: Save predictions
        print("→ Saving predictions...")
        pred_spark = save_predictions(
            pred_spark,
            cfg,
            model_name,
            mlflow.active_run().info.run_id,
            model.best_iteration_,
            IS_DATABRICKS,
            config,
        )

        # Step 9: Log feature importance
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        mlflow.log_table(
            feature_importance.head(20), f"feature_importance_{model_name}.json"
        )

        print(f"✓ Model training complete for: {model_name}\n")

        return {
            "model": model,
            "predictions": pred_spark,
            "feature_importance": feature_importance,
        }


def train_models_by_group(df_feat, cfg):
    """Train separate models based on strategy (global, by_family, by_family_group)"""

    strategy = cfg["model_strategy"]["type"]
    results = {}

    # Define model configurations based on strategy
    if strategy == "global":
        model_configs = [("global", df_feat)]

    elif strategy == "by_family":
        families = [row.family for row in df_feat.select("family").distinct().collect()]
        model_configs = [
            (family, df_feat.filter(F.col("family") == family)) for family in families
        ]

    elif strategy == "by_family_group":
        model_configs = [
            (group_name, df_feat.filter(F.col("family").isin(families)))
            for group_name, families in cfg["model_strategy"]["family_groups"].items()
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Train each model
    print(f"\nTraining {len(model_configs)} model(s) using strategy: {strategy}")
    for model_name, df_subset in model_configs:
        results[model_name] = train_single_model(df_subset, cfg, model_name)

    return results


# ========================================
# MAIN EXECUTION
# ========================================

from datetime import timedelta


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/daniel.more.torres@gmail.com/favorita_lgbm_regressor")

# Compute cutoff date
date_col = cfg["data"]["date_col"]
print(date_col)
max_date = df_raw.select(F.max(date_col)).collect()[0][0]
print(max_date)
cutoff_date = max_date - timedelta(days=180)
print(cutoff_date)

df_raw_train = df_raw.filter(F.col(date_col) <= F.lit(cutoff_date))
df_raw_test  = df_raw.filter(F.col(date_col) >  F.lit(cutoff_date))


mlflow.end_run()


with mlflow.start_run(run_name="favorita_lgbm_multi_model"):
    

    mlflow.log_param("strategy", cfg["model_strategy"]["type"])
    mlflow.log_params(cfg["model"]["params"])
    mlflow.log_param("config", str(cfg))

    # Step 1: Build Features
    print("\n" + "=" * 60)
    print("STEP 1: Building Features")
    print("=" * 60)
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
    print(f"✓ Features built: {len(df_feat.columns)} columns")

    # Create train/test

    df_feat_train = df_raw.filter(F.col(date_col) <= F.lit(cutoff_date))
    df_feat_test  = df_raw.filter(F.col(date_col) >  F.lit(cutoff_date))

    train_end_date = (
    df_raw_train
    .select(F.max(F.col(date_col)).alias("max_date"))
    .collect()[0]["max_date"]
    )

    test_start_date = (
        df_raw_test
        .select(F.min(F.col(date_col)).alias("min_date"))
        .collect()[0]["min_date"]
    )
    # Log split info
    mlflow.log_param("test_days", 180)
    mlflow.log_param("train_end_date", str(train_end_date))
    mlflow.log_param("test_start_date", str(test_start_date))

    # Step 2: Train models based on strategy
    print("\n" + "=" * 60)
    print("STEP 2: Training Models")
    print("=" * 60)
    all_results = train_models_by_group(df_feat_train, cfg)


    # Step 3: Compare results (if multiple models)
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("STEP 3: Model Comparison")
        print("=" * 60)
        comparison = {}
        for model_name, result in all_results.items():
            comparison[model_name] = {
                "num_features": len(result["feature_importance"]),
                "best_iteration": result["model"].best_iteration_,
            }
            print(f"  {model_name}: {comparison[model_name]}")

        comparison_df = pd.DataFrame(comparison).T
        mlflow.log_table(comparison_df, "model_comparison.json")

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
