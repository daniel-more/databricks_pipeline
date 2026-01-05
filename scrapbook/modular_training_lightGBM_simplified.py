!pip install -q mlflow lightgbm
!pip install icecream
%restart_python
# %%
import os
from datetime import datetime, timedelta
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

# add timezone
spark.conf.set("spark.sql.session.timeZone", "America/New_York")


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
        "val_horizon": 30,  # NEW: validation set size
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
# HELPER FUNCTIONS
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
    num_iteration = model.best_iteration_ if model.best_iteration_ > 0 else None
    predictions_log = model.predict(X_test, num_iteration=num_iteration)
    predictions = np.expm1(predictions_log)

    return predictions


def train_test_val_split(
    df,
    date_col,
    group_cols,
    test_horizon,
    val_horizon,
    min_train_periods=56
):
    """
    Split data into train/validation/test sets using time-based splitting.
    
    Parameters:
    -----------
    df : pyspark DataFrame
        Input dataframe with time series data
    date_col : str
        Name of date column
    group_cols : list
        Grouping columns (e.g., ['family', 'store_nbr'])
    test_horizon : int
        Number of days for test set (holdout)
    val_horizon : int
        Number of days for validation set
    min_train_periods : int
        Minimum number of training periods required
    
    Returns:
    --------
    train_spark, val_spark, test_spark, split_info : tuple
    """
    # Get the maximum date in the dataset
    max_date = df.select(F.max(date_col)).collect()[0][0]
    
    # Calculate split dates
    test_start_date = max_date - timedelta(days=test_horizon - 1)
    val_start_date = test_start_date - timedelta(days=val_horizon)
    train_end_date = val_start_date - timedelta(days=1)
    
    print(f"\nSplit dates:")
    print(f"  Train: up to {train_end_date}")
    print(f"  Validation: {val_start_date} to {test_start_date - timedelta(days=1)}")
    print(f"  Test: {test_start_date} to {max_date}")
    
    # Split the data
    train_spark = df.filter(F.col(date_col) <= F.lit(train_end_date))
    val_spark = df.filter(
        (F.col(date_col) >= F.lit(val_start_date)) & 
        (F.col(date_col) < F.lit(test_start_date))
    )
    test_spark = df.filter(F.col(date_col) >= F.lit(test_start_date))
    
    # Verify minimum training periods per group
    train_counts = (
        train_spark
        .groupBy(*group_cols)
        .agg(F.count("*").alias("count"))
        .filter(F.col("count") < min_train_periods)
    )
    
    if train_counts.count() > 0:
        print(f"Warning: {train_counts.count()} groups have fewer than {min_train_periods} training periods")
    
    # Print split statistics
    train_count = train_spark.count()
    val_count = val_spark.count()
    test_count = test_spark.count()
    total_count = train_count + val_count + test_count
    
    print(f"\nSplit statistics:")
    print(f"  Train: {train_count:,} rows ({train_count/total_count*100:.1f}%)")
    print(f"  Validation: {val_count:,} rows ({val_count/total_count*100:.1f}%)")
    print(f"  Test: {test_count:,} rows ({test_count/total_count*100:.1f}%)")
    print(f"  Total: {total_count:,} rows")
    
    return train_spark, val_spark, test_spark, {
        'train_end_date': train_end_date,
        'val_start_date': val_start_date,
        'test_start_date': test_start_date,
        'max_date': max_date
    }


def compute_metrics(y_true, y_pred, prefix=""):
    """Compute regression metrics"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}mape": mape
    }
    
    return metrics


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

        try:
            table_schema = spark.table(table_name).schema
            onpromotion_type = [f.dataType for f in table_schema if f.name == "onpromotion"][0]
            print(onpromotion_type)
            # Cast 'onpromotion' in pred_spark to match the Delta table type
            pred_spark = pred_spark.withColumn(
                "onpromotion",
                F.col("onpromotion").cast(onpromotion_type.simpleString()))
        except:
            print('Table not found')

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
# MAIN TRAINING FUNCTION
# ========================================

def train_single_model(df_feat, cfg, model_name):
    """Train and evaluate a single LightGBM model with proper train/val/test split"""

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

        # Step 1: Split into train/val/test with proper horizons
        print("→ Splitting train/validation/test data...")
        train_spark, val_spark, test_spark, split_info = train_test_val_split(
            df_feat,
            date_col=cfg["data"]["date_col"],
            group_cols=cfg["data"]["group_cols"],
            test_horizon=cfg["split"]["test_horizon"],
            val_horizon=cfg["split"]["val_horizon"],
            min_train_periods=cfg["data"]["min_train_periods"],
        )
        
        # Log split dates
        mlflow.log_param("train_end_date", str(split_info['train_end_date']))
        mlflow.log_param("val_start_date", str(split_info['val_start_date']))
        mlflow.log_param("test_start_date", str(split_info['test_start_date']))
        mlflow.log_param("val_horizon_days", cfg["split"]["val_horizon"])
        mlflow.log_param("test_horizon_days", cfg["split"]["test_horizon"])

        # Step 2: Convert to Pandas
        print("→ Converting to Pandas...")
        train_pdf = train_spark.toPandas()
        val_pdf = val_spark.toPandas()
        test_pdf = test_spark.toPandas()
        print(f"  Train size: {len(train_pdf):,}")
        print(f"  Val size: {len(val_pdf):,}")
        print(f"  Test size: {len(test_pdf):,}")

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
        train_pdf, _, feature_cols = prepare_dataframes(
            train_pdf,
            val_pdf,
            categorical_cols,
            numeric_cols,
            cfg["data"]["target_col"], 
            cfg["data"]["date_col"]
        )
        
        # Also prepare validation and test with same schema
        val_pdf, _, _ = prepare_dataframes(
            val_pdf,
            val_pdf,
            categorical_cols,
            numeric_cols,
            cfg["data"]["target_col"], 
            cfg["data"]["date_col"]
        )
        
        test_pdf, _, _ = prepare_dataframes(
            test_pdf,
            test_pdf,
            categorical_cols,
            numeric_cols,
            cfg["data"]["target_col"], 
            cfg["data"]["date_col"]
        )

        # Step 5: Train model with validation set for early stopping
        print("→ Training LightGBM model...")
        model = train_lgbm_model(
            train_pdf,
            val_pdf,  # Use proper validation set, NOT test set!
            feature_cols=feature_cols,
            target_col=cfg["data"]["target_col"],
            params=cfg["model"]["params"],
            early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
        )
        print(f"  Best iteration: {model.best_iteration_}")
        mlflow.log_metric("best_iteration", model.best_iteration_)

        # Step 6: Generate predictions on train, validation, and test sets
        print("→ Generating predictions...")
        
        # Training predictions (for overfitting check)
        train_predictions = predict_lgbm_model(model, train_pdf, feature_cols)
        train_pdf["prediction"] = train_predictions
        train_pdf["y"] = train_pdf[cfg["data"]["target_col"]]
        train_pdf["split"] = "train"
        
        # Validation predictions (used for early stopping)
        val_predictions = predict_lgbm_model(model, val_pdf, feature_cols)
        val_pdf["prediction"] = val_predictions
        val_pdf["y"] = val_pdf[cfg["data"]["target_col"]]
        val_pdf["split"] = "validation"
        
        # Test predictions (final holdout evaluation)
        test_predictions = predict_lgbm_model(model, test_pdf, feature_cols)
        test_pdf["prediction"] = test_predictions
        test_pdf["y"] = test_pdf[cfg["data"]["target_col"]]
        test_pdf["split"] = "test"
        
        print(f"  Train predictions: {len(train_predictions):,}")
        print(f"  Validation predictions: {len(val_predictions):,}")
        print(f"  Test predictions: {len(test_predictions):,}")

        # Step 7: Calculate and log metrics for all three sets
        print("→ Computing metrics...")
        
        # Training metrics
        train_metrics = compute_metrics(train_pdf["y"], train_pdf["prediction"], prefix="train_")
        for k, v in train_metrics.items():
            mlflow.log_metric(k, v)
        print(f"  Train RMSE: {train_metrics['train_rmse']:.2f}, MAE: {train_metrics['train_mae']:.2f}, MAPE: {train_metrics['train_mape']:.2f}%")
        
        # Validation metrics
        val_metrics = compute_metrics(val_pdf["y"], val_pdf["prediction"], prefix="val_")
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v)
        print(f"  Val RMSE: {val_metrics['val_rmse']:.2f}, MAE: {val_metrics['val_mae']:.2f}, MAPE: {val_metrics['val_mape']:.2f}%")
        
        # Test metrics (final evaluation)
        test_metrics = compute_metrics(test_pdf["y"], test_pdf["prediction"], prefix="test_")
        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)
        print(f"  Test RMSE: {test_metrics['test_rmse']:.2f}, MAE: {test_metrics['test_mae']:.2f}, MAPE: {test_metrics['test_mape']:.2f}%")

        # Check for overfitting
        if train_metrics['train_rmse'] < val_metrics['val_rmse'] * 0.7:
            print(f"  ⚠️  Warning: Possible overfitting detected (train RMSE much lower than val)")
        
        # Step 8: Combine all predictions for saving
        combined_predictions = pd.concat([train_pdf, val_pdf, test_pdf], ignore_index=True)
        pred_spark = spark.createDataFrame(combined_predictions)

        # Step 9: Log model with signature
        print("→ Logging model to MLflow...")
        from mlflow.models.signature import infer_signature

        signature = infer_signature(
            train_pdf[feature_cols], model.predict(train_pdf[feature_cols])
        )
        mlflow.lightgbm.log_model(
            model, name=f"model_{model_name}", signature=signature
        )

        # Step 10: Save predictions (train, val, and test)
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

        # Step 11: Log feature importance
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        mlflow.log_table(
            feature_importance.head(20), f"feature_importance_{model_name}.json"
        )
        
        # Step 12: Create and log prediction analysis
        print("→ Creating prediction analysis...")
        
        # Error analysis by date and split
        error_by_date = (
            combined_predictions
            .groupby([cfg["data"]["date_col"], "split"])
            .agg({
                "y": "sum",
                "prediction": "sum"
            })
            .reset_index()
        )
        error_by_date["error"] = error_by_date["y"] - error_by_date["prediction"]
        error_by_date["abs_error"] = np.abs(error_by_date["error"])
        error_by_date["pct_error"] = (error_by_date["error"] / (error_by_date["y"] + 1e-10)) * 100
        
        mlflow.log_table(error_by_date, f"error_analysis_{model_name}.json")
        
        # Summary statistics
        summary_stats = combined_predictions.groupby("split").agg({
            "y": ["mean", "std", "min", "max"],
            "prediction": ["mean", "std", "min", "max"]
        }).reset_index()
        
        mlflow.log_table(summary_stats, f"summary_stats_{model_name}.json")

        print(f"✓ Model training complete for: {model_name}\n")

        return {
            "model": model,
            "predictions": pred_spark,
            "feature_importance": feature_importance,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
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
from pyspark.sql.functions import col
df_raw = df_raw.filter(col("family") == "BEVERAGES")


from pyspark.sql.functions import col

# Filter to single family for testing (remove this line for full training)
# df_raw = df_raw.filter(col("family") == "BEVERAGES")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/daniel.more.torres@gmail.com/favorita_lgbm_regressor")

<<<<<<< Updated upstream
# End any existing runs
=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    # Log dataset info
    date_col = cfg["data"]["date_col"]
    max_date = df_feat.select(F.max(date_col)).collect()[0][0]
    min_date = df_feat.select(F.min(date_col)).collect()[0][0]
    
    mlflow.log_param("data_start_date", str(min_date))
    mlflow.log_param("data_end_date", str(max_date))
    mlflow.log_param("total_rows", df_feat.count())
=======
 >>>>>>> Stashed changes

    # Step 2: Train models based on strategy
    # The train/val/test split happens INSIDE train_single_model()
    print("\n" + "=" * 60)
    print("STEP 2: Training Models with Train/Val/Test Split")
    print("=" * 60)
    all_results = train_models_by_group(df_feat, cfg)

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
                "train_rmse": result["train_metrics"]["train_rmse"],
                "val_rmse": result["val_metrics"]["val_rmse"],
                "test_rmse": result["test_metrics"]["test_rmse"],
            }
            print(f"  {model_name}: {comparison[model_name]}")

        comparison_df = pd.DataFrame(comparison).T
        mlflow.log_table(comparison_df, "model_comparison.json")

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)