# ========================================
# Gold Layer: Sales + Forecast Delta Table
# ========================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import yaml

spark = SparkSession.builder.getOrCreate()
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration
CATALOG = config["databricks"]["catalog"]
SCHEMA = config["databricks"]["schema"]

SILVER_TRAINING_TABLE = f"{CATALOG}.{SCHEMA}.silver_training"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.lightgbm_silver_test_predictions"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_sales_forecast"


# ========================================
# 1. CREATE GOLD TABLE SCHEMA
# ========================================

def create_gold_sales_forecast_table():
    """Create the gold layer table with proper schema and partitioning"""
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {GOLD_TABLE} (
        date DATE NOT NULL,
        family STRING NOT NULL,
        store_nbr BIGINT NOT NULL,
        sales DOUBLE,                          -- Actual sales (NULL for future forecasts)
        forecast DOUBLE,                       -- Predicted sales (NULL for historical actuals)
        record_type STRING NOT NULL,           -- 'actual', 'forecast', 'backtest'
        model_name STRING,                     -- Which model generated the forecast
        model_type STRING,                     -- e.g., 'lgbm'
        model_version STRING,                  -- MLflow run_id
        model_strategy STRING,                 -- 'global', 'by_family', etc.
        best_iteration INT,                    -- LightGBM best iteration
        prediction_timestamp TIMESTAMP,        -- When forecast was generated
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    USING DELTA
    PARTITIONED BY (date, model_name)
    COMMENT 'Gold layer table combining historical sales and forecasts'
    """
    
    spark.sql(create_table_sql)
    print(f"✓ Created table: {GOLD_TABLE}")


# ========================================
# 2. BACKFILL HISTORICAL ACTUALS
# ========================================

def backfill_historical_actuals(overwrite=False):
    """Load historical sales data as 'actual' records"""
    
    print("→ Loading historical sales data...")
    
    # Read silver training data
    df_actuals = spark.read.format("delta").table(SILVER_TRAINING_TABLE)
    
    # Transform to gold schema
    df_gold_actuals = (
        df_actuals
        .select(
            F.to_date("date").alias("date"),
            "family",
            F.col("store_nbr").cast("bigint").alias("store_nbr"),  # Cast to BIGINT
            "sales",
            F.lit(None).cast("double").alias("forecast"),
            F.lit("actual").alias("record_type"),
            F.lit(None).cast("string").alias("model_name"),
            F.lit(None).cast("string").alias("model_type"),
            F.lit(None).cast("string").alias("model_version"),
            F.lit(None).cast("string").alias("model_strategy"),
            F.lit(None).cast("int").alias("best_iteration"),
            F.lit(None).cast("timestamp").alias("prediction_timestamp"),
        )
        .withColumn("created_at", F.current_timestamp())
        .withColumn("updated_at", F.current_timestamp())
    )
    
    # Write to gold table
    mode = "overwrite" if overwrite else "append"
    
    df_gold_actuals.write.format("delta").mode(mode).partitionBy(
        "date", "model_name"
    ).saveAsTable(GOLD_TABLE)
    
    row_count = df_gold_actuals.count()
    print(f"✓ Inserted {row_count:,} historical actual records")
    
    return df_gold_actuals


# ========================================
# 3. APPEND FORECAST PREDICTIONS
# ========================================

def append_forecast_predictions(model_name_filter=None):
    """Append predictions from predictions table as 'forecast' records"""
    
    print("→ Loading forecast predictions...")
    
    # Read predictions
    df_predictions = spark.read.format("delta").table(PREDICTIONS_TABLE)
    
    # Optional filter by model name
    if model_name_filter:
        df_predictions = df_predictions.filter(F.col("model_name") == model_name_filter)
    
    # Transform to gold schema
    df_gold_forecasts = (
        df_predictions
        .select(
            F.to_date("date").alias("date"),
            "family",
            "store_nbr",
            F.lit(None).cast("double").alias("sales"),  # NULL for forecasts
            F.col("prediction").alias("forecast"),
            F.lit("forecast").alias("record_type"),
            "model_name",
            "model_type",
            "model_version",
            "model_strategy",
            "best_iteration",
            "prediction_timestamp",
        )
        .withColumn("created_at", F.current_timestamp())
        .withColumn("updated_at", F.current_timestamp())
    )
    
    # Append to gold table
    df_gold_forecasts.write.format("delta").mode("append").partitionBy(
        "date", "model_name"
    ).saveAsTable(GOLD_TABLE)
    
    row_count = df_gold_forecasts.count()
    print(f"✓ Inserted {row_count:,} forecast records")
    
    return df_gold_forecasts


# ========================================
# 4. APPEND BACKTEST PREDICTIONS
# ========================================

def append_backtest_predictions():
    """
    Append predictions on test set (historical period) as 'backtest' records.
    These are predictions made on historical data for model evaluation.
    """
    
    print("→ Loading backtest predictions...")
    
    df_predictions = spark.read.format("delta").table(PREDICTIONS_TABLE)
    
    # Get the actual sales for the test period to identify backtest records
    df_actuals = spark.read.format("delta").table(SILVER_TRAINING_TABLE)
    
    # Join predictions with actuals to identify backtest period
    # If actual sales exists for the prediction date, it's a backtest
    df_backtest = (
        df_predictions
        .join(
            df_actuals.select(
                "date", 
                "family", 
                F.col("store_nbr").cast("bigint").alias("store_nbr"),  # Cast to BIGINT
                "sales"
            ),
            on=["date", "family", "store_nbr"],
            how="inner"  # Only predictions that have actual sales
        )
        .select(
            F.to_date(df_predictions["date"]).alias("date"),
            df_predictions["family"],
            df_predictions["store_nbr"],  # Already BIGINT from predictions
            df_actuals["sales"],  # Actual sales
            F.col("prediction").alias("forecast"),  # Predicted sales
            F.lit("backtest").alias("record_type"),
            "model_name",
            "model_type",
            "model_version",
            "model_strategy",
            "best_iteration",
            "prediction_timestamp",
        )
        .withColumn("created_at", F.current_timestamp())
        .withColumn("updated_at", F.current_timestamp())
    )
    
    # Append to gold table
    df_backtest.write.format("delta").mode("append").partitionBy(
        "date", "model_name"
    ).saveAsTable(GOLD_TABLE)
    
    row_count = df_backtest.count()
    print(f"✓ Inserted {row_count:,} backtest records")
    
    return df_backtest


# ========================================
# 5. UPDATE EXISTING FORECASTS (MERGE)
# ========================================

def update_forecasts_for_model(model_name):
    """
    Update existing forecasts for a specific model using MERGE.
    Useful when retraining and want to replace old forecasts.
    """
    
    print(f"→ Updating forecasts for model: {model_name}")
    
    # Read new predictions
    df_new_predictions = (
        spark.read.format("delta")
        .table(PREDICTIONS_TABLE)
        .filter(F.col("model_name") == model_name)
    )
    
    # Transform to gold schema
    df_new_forecasts = (
        df_new_predictions
        .select(
            F.to_date("date").alias("date"),
            "family",
            "store_nbr",
            F.lit(None).cast("double").alias("sales"),
            F.col("prediction").alias("forecast"),
            F.lit("forecast").alias("record_type"),
            "model_name",
            "model_type",
            "model_version",
            "model_strategy",
            "best_iteration",
            "prediction_timestamp",
        )
        .withColumn("created_at", F.current_timestamp())
        .withColumn("updated_at", F.current_timestamp())
    )
    
    # Create temp view for merge
    df_new_forecasts.createOrReplaceTempView("new_forecasts")
    
    # MERGE statement
    merge_sql = f"""
    MERGE INTO {GOLD_TABLE} target
    USING new_forecasts source
    ON target.date = source.date
        AND target.family = source.family
        AND target.store_nbr = source.store_nbr
        AND target.model_name = source.model_name
        AND target.record_type = 'forecast'
    WHEN MATCHED THEN UPDATE SET
        target.forecast = source.forecast,
        target.model_version = source.model_version,
        target.model_strategy = source.model_strategy,
        target.best_iteration = source.best_iteration,
        target.prediction_timestamp = source.prediction_timestamp,
        target.updated_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT *
    """
    
    spark.sql(merge_sql)
    print(f"✓ Merged forecasts for model: {model_name}")


# ========================================
# 6. UTILITY VIEWS
# ========================================

def create_utility_views():
    """Create helpful views on top of gold table"""
    
    # View 1: Latest forecast per model
    latest_forecast_view = f"""
    CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.v_latest_forecasts AS
    WITH ranked_forecasts AS (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY date, family, store_nbr, model_name, record_type
                ORDER BY prediction_timestamp DESC
            ) as rn
        FROM {GOLD_TABLE}
        WHERE record_type = 'forecast'
    )
    SELECT * EXCEPT(rn)
    FROM ranked_forecasts
    WHERE rn = 1
    """
    spark.sql(latest_forecast_view)
    print(f"✓ Created view: v_latest_forecasts")
    
    # View 2: Combined actuals + latest forecasts
    combined_view = f"""
    CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.v_sales_and_forecasts AS
    SELECT 
        date,
        family,
        store_nbr,
        COALESCE(sales, forecast) as value,
        CASE 
            WHEN sales IS NOT NULL THEN 'actual'
            ELSE 'forecast'
        END as data_type,
        model_name,
        model_version,
        prediction_timestamp
    FROM {CATALOG}.{SCHEMA}.v_latest_forecasts
    
    UNION ALL
    
    SELECT
        date,
        family,
        store_nbr,
        sales as value,
        'actual' as data_type,
        NULL as model_name,
        NULL as model_version,
        NULL as prediction_timestamp
    FROM {GOLD_TABLE}
    WHERE record_type = 'actual'
    """
    spark.sql(combined_view)
    print(f"✓ Created view: v_sales_and_forecasts")
    
    # View 3: Forecast accuracy metrics
    accuracy_view = f"""
    CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.v_forecast_accuracy AS
    SELECT
        model_name,
        model_version,
        family,
        store_nbr,
        COUNT(*) as num_predictions,
        AVG(ABS(sales - forecast)) as mae,
        AVG(ABS(sales - forecast) / NULLIF(sales, 0)) as mape,
        SQRT(AVG(POW(sales - forecast, 2))) as rmse,
        MIN(prediction_timestamp) as first_prediction_time,
        MAX(prediction_timestamp) as last_prediction_time
    FROM {GOLD_TABLE}
    WHERE record_type = 'backtest'
        AND sales IS NOT NULL 
        AND forecast IS NOT NULL
    GROUP BY model_name, model_version, family, store_nbr
    """
    spark.sql(accuracy_view)
    print(f"✓ Created view: v_forecast_accuracy")


# ========================================
# 7. MAIN ORCHESTRATION FUNCTION
# ========================================

def build_gold_sales_forecast_table(
    create_table=True,
    backfill_actuals=True,
    append_forecasts=True,
    append_backtests=True,
    create_views=True,
    overwrite_actuals=False
):
    """
    Main function to build the complete gold sales + forecast table.
    
    Args:
        create_table: Create the Delta table schema
        backfill_actuals: Load historical sales data
        append_forecasts: Append forecast predictions
        append_backtests: Append backtest predictions (predictions on historical data)
        create_views: Create utility views
        overwrite_actuals: Whether to overwrite existing actuals
    """
    
    print("\n" + "="*60)
    print("BUILDING GOLD SALES + FORECAST TABLE")
    print("="*60)
    
    # Step 1: Create table
    if create_table:
        print("\n→ Step 1: Creating table schema...")
        create_gold_sales_forecast_table()
    
    # Step 2: Backfill actuals
    if backfill_actuals:
        print("\n→ Step 2: Backfilling historical actuals...")
        backfill_historical_actuals(overwrite=overwrite_actuals)
    
    # Step 3: Append backtests (predictions on historical period)
    if append_backtests:
        print("\n→ Step 3: Appending backtest predictions...")
        append_backtest_predictions()
    
    # Step 4: Append forecasts (predictions on future period)
    if append_forecasts:
        print("\n→ Step 4: Appending forecast predictions...")
        append_forecast_predictions()
    
    # Step 5: Create views
    if create_views:
        print("\n→ Step 5: Creating utility views...")
        create_utility_views()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    summary = spark.sql(f"""
        SELECT 
            record_type,
            model_name,
            COUNT(*) as record_count,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM {GOLD_TABLE}
        GROUP BY record_type, model_name
        ORDER BY record_type, model_name
    """)
    
    summary.show(truncate=False)
    
    print("\n✓ Gold table build complete!")
    print(f"   Table: {GOLD_TABLE}")
    print(f"   Views: v_latest_forecasts, v_sales_and_forecasts, v_forecast_accuracy")
    

# ========================================
# 8. EXAMPLE QUERIES
# ========================================

def example_queries():
    """Example queries to demonstrate gold table usage"""
    
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)
    
    # Query 1: Compare actual vs forecast for specific family
    print("\n1. Actual vs Forecast for BEVERAGES:")
    query1 = f"""
    SELECT 
        date,
        store_nbr,
        sales as actual,
        forecast,
        ABS(sales - forecast) as absolute_error,
        ABS(sales - forecast) / NULLIF(sales, 0) * 100 as pct_error
    FROM {GOLD_TABLE}
    WHERE family = 'BEVERAGES'
        AND record_type = 'backtest'
        AND store_nbr = 1
    ORDER BY date DESC
    LIMIT 10
    """
    spark.sql(query1).show()
    
    # Query 2: Future forecasts
    print("\n2. Future Forecasts (next 7 days):")
    query2 = f"""
    SELECT 
        date,
        family,
        store_nbr,
        forecast,
        model_name,
        model_version
    FROM {CATALOG}.{SCHEMA}.v_latest_forecasts
    WHERE record_type = 'forecast'
        AND date >= CURRENT_DATE()
    ORDER BY date, family, store_nbr
    LIMIT 20
    """
    spark.sql(query2).show()
    
    # Query 3: Model performance comparison
    print("\n3. Model Performance Comparison:")
    query3 = f"""
    SELECT 
        model_name,
        family,
        ROUND(AVG(mae), 2) as avg_mae,
        ROUND(AVG(mape) * 100, 2) as avg_mape_pct,
        ROUND(AVG(rmse), 2) as avg_rmse,
        SUM(num_predictions) as total_predictions
    FROM {CATALOG}.{SCHEMA}.v_forecast_accuracy
    GROUP BY model_name, family
    ORDER BY model_name, family
    """
    spark.sql(query3).show()


# ========================================
# EXECUTION
# ========================================

if __name__ == "__main__":
    # Step 0: Diagnose the data type issue
    print("\n" + "="*60)
    print("DIAGNOSING DATA TYPES")
    print("="*60)
    
    # Check silver training table schema
    print("\n1. Silver Training Table Schema:")
    df_silver = spark.read.format("delta").table(SILVER_TRAINING_TABLE)
    df_silver.select("store_nbr", "sales").printSchema()
    print(f"   store_nbr type: {dict(df_silver.dtypes)['store_nbr']}")
    
    # Check predictions table schema
    print("\n2. Predictions Table Schema:")
    df_pred = spark.read.format("delta").table(PREDICTIONS_TABLE)
    df_pred.select("store_nbr", "prediction").printSchema()
    print(f"   store_nbr type: {dict(df_pred.dtypes)['store_nbr']}")
    
    # Drop existing table completely
    print(f"\n3. Dropping existing table: {GOLD_TABLE}")
    spark.sql(f"DROP TABLE IF EXISTS {GOLD_TABLE}")
    
    # Also remove the underlying data files if running locally
    try:
        spark.sql(f"VACUUM {GOLD_TABLE} RETAIN 0 HOURS")
    except:
        print("   (Vacuum skipped - table doesn't exist)")
    
    # Build the complete gold table
    build_gold_sales_forecast_table(
        create_table=True,
        backfill_actuals=True,
        append_forecasts=True,
        append_backtests=True,
        create_views=True,
        overwrite_actuals=False  # Set to True for fresh start
    )
    
    # Run example queries
    example_queries()
    
    # Optional: Update specific model forecasts
    # update_forecasts_for_model("global")