# PySpark reusable forecasting framework
from typing import Any, Dict, List, Tuple

from icecream import ic
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


def aggregate_to_granularity(
    df,
    date_col: str,
    target_col: str,
    group_cols: list,
    agg: str = "sum",
    extra_numeric_aggs: dict = None,
):
    """
    Aggregates df to exactly one row per (group_cols, date_col).
    - agg: aggregation for the target ('sum' | 'mean' | 'max' | 'min')
    - extra_numeric_aggs: optional dict {col_name: 'sum'|'mean'|...} for other drivers
      (e.g., promo_spend, price) when using global models with exogenous features.
    """
    agg_expr = {
        "sum": F.sum
        # "mean": F.avg,
        # "avg": F.avg,
        # "max": F.max,
        # "min": F.min
    }.get(agg.lower())
    if agg_expr is None:
        raise ValueError(f"Unsupported agg='{agg}'")

    # Base target aggregation
    metrics = [agg_expr(F.col(target_col)).alias(target_col)]

    # Optional aggregations for other numeric columns
    if extra_numeric_aggs:
        for c, a in extra_numeric_aggs.items():
            fun = {
                "sum": F.sum,
                "mean": F.avg,
                "avg": F.avg,
                "max": F.max,
                "min": F.min,
            }.get(a.lower())
            if fun is None:
                raise ValueError(f"Unsupported agg '{a}' for column '{c}'")
            metrics.append(fun(F.col(c)).alias(c))

    # Group to one row per (group_cols, date)
    gdf = (
        df.groupBy(*group_cols, date_col)
        .agg(*metrics)
        .withColumn(date_col, F.to_date(F.col(date_col)))
    )
    return gdf


# ----------------------------
# 1) Feature engineering
# ----------------------------
def build_features(
    df: DataFrame,
    date_col: str,
    target_col: str,
    group_cols: List[str],
    lags: List[int],
    mas: List[int],
    add_time_signals: bool = True,
    pre_aggregate: bool = True,
    target_agg: str = "sum",
    extra_numeric_aggs: Dict[str, str] = None,
) -> DataFrame:
    """
    Creates lag and moving average features for time series forecasting.
    - Ensures one row per (group_cols, date) before applying window functions.
    - Avoids nesting window functions inside groupBy aggregates.
    """

    # Step 1: Pre-aggregate if needed
    if pre_aggregate:
        agg_funcs = {
            "sum": F.sum,
            "mean": F.avg,
            "avg": F.avg,
            "max": F.max,
            "min": F.min,
        }
        if target_agg.lower() not in agg_funcs:
            raise ValueError(f"Unsupported aggregation: {target_agg}")
        metrics = [agg_funcs[target_agg.lower()](F.col(target_col)).alias(target_col)]
        if extra_numeric_aggs:
            for col, agg in extra_numeric_aggs.items():
                if agg.lower() not in agg_funcs:
                    raise ValueError(
                        f"Unsupported aggregation '{agg}' for column '{col}'"
                    )
                metrics.append(agg_funcs[agg.lower()](F.col(col)).alias(col))
        df = (
            df.groupBy(*group_cols, date_col)
            .agg(*metrics)
            .withColumn(date_col, F.to_date(F.col(date_col)))
        )

    w = Window.partitionBy(*group_cols).orderBy(F.col(date_col))

    # Lag features
    for L in lags:
        df = df.withColumn(f"lag_{L}", F.lag(F.col(target_col), L).over(w))

    # Moving averages
    for M in mas:
        df = df.withColumn(
            f"ma_{M}", F.avg(F.col(target_col)).over(w.rowsBetween(-M + 1, 0))
        )

    # Calendar/time signals (scale‑independent)
    if add_time_signals:
        df = (
            df.withColumn("dow", F.dayofweek(F.col(date_col)))  # 1..7
            .withColumn("dom", F.dayofmonth(F.col(date_col)))
            .withColumn("weekofyear", F.weekofyear(F.col(date_col)))
            .withColumn("month", F.month(F.col(date_col)))
            .withColumn("year", F.year(F.col(date_col)))
        )
    # Drop rows where lags/moving averages are missing
    max_warmup = max(lags + mas) if (lags or mas) else 0
    if max_warmup > 0:
        df = (
            df.withColumn("_row_num", F.row_number().over(w))
            .filter(F.col("_row_num") > max_warmup)
            .drop("_row_num")
        )

    return df


# ----------------------------
# 2) Train/Test split
# ----------------------------
def train_test_split(
    df: DataFrame,
    date_col: str,
    group_cols: List[str],
    mode: str = "horizon",
    train_end_date: str = "",
    test_horizon: int = 28,
    min_train_periods: int = 56,
) -> Tuple[DataFrame, DataFrame]:
    # ensure enough history per group
    counts = df.groupBy(*group_cols).agg(F.count("*").alias("n"))
    valid_groups = counts.filter(
        F.col("n") >= (min_train_periods + (test_horizon if mode == "horizon" else 0))
    ).select(*group_cols)
    df = df.join(valid_groups, on=group_cols, how="inner")

    if mode == "date" and train_end_date:
        train = df.filter(F.col(date_col) <= F.to_date(F.lit(train_end_date)))
        test = df.filter(F.col(date_col) > F.to_date(F.lit(train_end_date)))
    else:
        # horizon: last H periods are test per group
        w = (
            Window.partitionBy(*group_cols)
            .orderBy(F.col(date_col))
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
        idx_df = df.withColumn(
            "rnk",
            F.row_number().over(
                Window.partitionBy(*group_cols).orderBy(F.col(date_col))
            ),
        )
        last_df = idx_df.groupBy(*group_cols).agg(F.max("rnk").alias("max_rnk"))
        df = idx_df.join(last_df, on=group_cols)
        test = df.filter(F.col("rnk") > (F.col("max_rnk") - F.lit(test_horizon))).drop(
            "rnk", "max_rnk"
        )
        train = df.filter(
            F.col("rnk") <= (F.col("max_rnk") - F.lit(test_horizon))
        ).drop("rnk", "max_rnk")

    return train, test


# ----------------------------
# 3) Model factory (global & local)
# ----------------------------
def model_factory(model_type: str, params: Dict[str, Any]):
    if model_type == "spark_gbt":
        return GBTRegressor(featuresCol="features", labelCol="label", **params)
    elif model_type == "spark_rf":
        return RandomForestRegressor(featuresCol="features", labelCol="label", **params)
    elif model_type == "glm":
        return LinearRegression(featuresCol="features", labelCol="label", **params)
    elif model_type in {"prophet_local", "arima_local"}:
        # placeholder: implemented via pandas UDF below
        return model_type
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------
# 4) Assemble pipeline (global models)
# ----------------------------
def assemble_global_pipeline(
    df: DataFrame,
    target_col: str,
    group_cols: List[str],
    categorical_cols: List[str],
    feature_cols: List[str],
    estimator,
) -> Pipeline:
    # StringIndex + OHE for categorical group columns
    stages = []
    indexed_cols = []
    for c in categorical_cols:
        si = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_ohe"])
        stages += [si, ohe]
        indexed_cols.append(f"{c}_ohe")

        ic(indexed_cols)
        ic(feature_cols)

    assembler = VectorAssembler(
        inputCols=feature_cols + indexed_cols, outputCol="features"
    )
    # Label column
    # df = df.withColumn("label", F.col(target_col).cast("double"))
    stages += [assembler, estimator]
    return Pipeline(stages=stages)


# ----------------------------
# 5) Local models via pandas UDF (Prophet / ARIMA)
# ----------------------------
def forecast_local_udf(
    model_type: str, horizon: int, freq: str, date_col: str, target_col: str
):
    """
    Returns a grouped map pandas UDF to train a local model per group and forecast `horizon` steps.
    Requires 'prophet' or 'statsmodels' installed on cluster.
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        DateType,
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    out_schema = T.StructType(
        [
            *(
                T.StructField(c, T.StringType(), False) for c in []
            ),  # placeholder—group cols will be appended dynamically
            T.StructField(date_col, DateType(), False),
            T.StructField("yhat", DoubleType(), False),
        ]
    )

    # We’ll construct schema dynamically at callsite; here we return the closure
    @pandas_udf(
        "date yhat double", F.PandasUDFType.GROUPED_MAP
    )  # simplified signature; we will build per call
    def _udf(pdf: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        # Identify group columns (all non date/target columns are treated as keys)
        key_cols = [c for c in pdf.columns if c not in {date_col, target_col}]
        gvals = pdf.iloc[0][key_cols].to_dict()

        # Train model
        if model_type == "prophet_local":
            from prophet import Prophet

            m = Prophet()
            df_fit = pdf[[date_col, target_col]].rename(
                columns={date_col: "ds", target_col: "y"}
            )
            m.fit(df_fit)
            future = m.make_future_dataframe(periods=horizon, freq=freq)
            yhat = (
                m.predict(future)
                .tail(horizon)[["ds", "yhat"]]
                .rename(columns={"ds": date_col})
            )
        elif model_type == "arima_local":
            import statsmodels.api as sm

            y = pdf[target_col].astype(float)
            # Simple ARIMA(1,1,1) example; customize as needed
            model = sm.tsa.ARIMA(y, order=(1, 1, 1)).fit()
            fc = model.forecast(steps=horizon)
            yhat = pd.DataFrame(
                {
                    date_col: pd.date_range(
                        pdf[date_col].max(), periods=horizon + 1, freq=freq
                    )[1:],
                    "yhat": fc,
                }
            )
        else:
            raise ValueError("Unsupported local model")

        for k, v in gvals.items():
            yhat[k] = v
        return yhat[[*key_cols, date_col, "yhat"]]

    return _udf


# ----------------------------
# 6) Metrics: wMAPE, sMAPE, MASE
# ----------------------------

from typing import List, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def compute_metrics(
    pred_df: DataFrame,
    date_col: str,
    actual_col: str,  # name of the actuals column (e.g., "y")
    pred_col: str,  # name of the predictions column (e.g., "prediction")
    group_cols: List[str],  # e.g., ["family"] or ["family", "store_nbr"]
    mase_seasonality: int = 7,
    epsilon: float = 1e-12,  # numerical stability for denominators
) -> Tuple[DataFrame, DataFrame]:
    """
    Computes sMAPE, wMAPE, and MASE at both per-series and portfolio level.
    - Materializes window-based columns (lag for naive baseline) via withColumn,
      then aggregates (no window functions inside agg).
    - sMAPE: mean over time of symmetric absolute percentage error.
    - wMAPE: sum(|y - yhat|) / sum(y) at series and portfolio levels.
    - MASE : MAE(forecast) / MAE(seasonal-naive), with seasonality lag.

    Parameters
    ----------
    pred_df : DataFrame
        Must contain [date_col, actual_col, pred_col] and group_cols.
        Typically built as: predictions.join(actuals, on=[date]+group_cols).
    date_col : str
        The timestamp column.
    actual_col : str
        Actual values column name (e.g., "y").
    pred_col : str
        Predictions column name (e.g., "prediction").
    group_cols : List[str]
        Series identifiers.
    mase_seasonality : int
        Seasonal lag (e.g., 7 for weekly seasonality on daily data).
    epsilon : float
        Small constant to stabilize denominators.

    Returns
    -------
    by_series : DataFrame
        Columns: group_cols + [sMAPE, wMAPE, MASE]
    portfolio : DataFrame
        Columns: [sMAPE, wMAPE, MASE, level="portfolio"]
    """

    # --- 1) Basic hygiene: ensure numeric types and non-null predictions/actuals ---
    df = pred_df.withColumn(actual_col, F.col(actual_col).cast("double")).withColumn(
        pred_col, F.col(pred_col).cast("double")
    )

    # If needed, you can fill small nulls; usually predictions/actuals shouldn't be null.
    # df = df.na.fill({actual_col: 0.0, pred_col: 0.0})

    # --- 2) Materialize window-based naive baseline (seasonal lag) ---
    w = Window.partitionBy(*group_cols).orderBy(F.col(date_col))
    df = df.withColumn("naive", F.lag(F.col(actual_col), mase_seasonality).over(w))

    # --- 3) Error columns (computed row-wise, no window inside agg) ---
    # sMAPE per-row
    df = df.withColumn(
        "smape_row",
        F.when(
            (F.abs(F.col(actual_col)) + F.abs(F.col(pred_col))) > 0,
            F.abs(F.col(actual_col) - F.col(pred_col))
            / ((F.abs(F.col(actual_col)) + F.abs(F.col(pred_col))) / 2.0),
        ).otherwise(F.lit(0.0)),
    )

    # Absolute error for wMAPE
    df = df.withColumn("abs_err", F.abs(F.col(actual_col) - F.col(pred_col)))

    # MAE components for MASE
    df = df.withColumn("mae_forecast_row", F.abs(F.col(actual_col) - F.col(pred_col)))
    df = df.withColumn("mae_naive_row", F.abs(F.col(actual_col) - F.col("naive")))

    # --- 4) Per-series aggregation ---
    # Add safe denominators to handle zero/near-zero totals
    by_series = df.groupBy(*group_cols).agg(
        F.avg("smape_row").alias("sMAPE"),
        (F.sum("abs_err") / F.greatest(F.sum(F.col(actual_col)), F.lit(epsilon))).alias(
            "wMAPE"
        ),
        (
            F.avg("mae_forecast_row")
            / F.greatest(F.avg("mae_naive_row"), F.lit(epsilon))
        ).alias("MASE"),
    )

    # --- 5) Portfolio (overall) aggregation ---
    portfolio = df.agg(
        F.avg("smape_row").alias("sMAPE"),
        (F.sum("abs_err") / F.greatest(F.sum(F.col(actual_col)), F.lit(epsilon))).alias(
            "wMAPE"
        ),
        (
            F.avg("mae_forecast_row")
            / F.greatest(F.avg("mae_naive_row"), F.lit(epsilon))
        ).alias("MASE"),
    ).withColumn("level", F.lit("portfolio"))

    return by_series, portfolio


# ----------------------------
# 7) Rolling backtest
# ----------------------------
def rolling_backtest(
    df: DataFrame,
    date_col: str,
    target_col: str,
    group_cols: List[str],
    feature_params: Dict[str, Any],
    model_type: str,
    model_params: Dict[str, Any],
    folds: int,
    fold_horizon: int,
    step: int,
    mase_seasonality: int,
) -> DataFrame:
    results = []
    # Determine fold start dates per group
    # (We compute indices with row_number and slide window)
    w = Window.partitionBy(*group_cols).orderBy(F.col(date_col))
    idx = df.withColumn("rnk", F.row_number().over(w))
    max_idx = idx.groupBy(*group_cols).agg(F.max("rnk").alias("max_rnk"))

    for k in range(folds):
        cut = idx.join(max_idx, on=group_cols).withColumn(
            "cutoff", F.col("max_rnk") - F.lit(fold_horizon + k * step)
        )
        train_k = cut.filter(F.col("rnk") <= F.col("cutoff")).drop(
            "rnk", "max_rnk", "cutoff"
        )
        test_k = cut.filter(F.col("rnk") > F.col("cutoff")).drop(
            "rnk", "max_rnk", "cutoff"
        )

        train_k = build_features(
            train_k, date_col, target_col, group_cols, **feature_params
        )
        test_k = build_features(
            test_k, date_col, target_col, group_cols, **feature_params
        )

        if model_type.endswith("_local"):
            # Local model per group via pandas UDF
            udf = forecast_local_udf(
                model_type,
                fold_horizon,
                feature_params.get("freq", "D"),
                date_col,
                target_col,
            )
            pred_k = (
                train_k.groupBy(*group_cols)
                .apply(udf)
                .join(
                    test_k.select(*group_cols, date_col, target_col),
                    on=group_cols + [date_col],
                    how="inner",
                )
                .withColumnRenamed("yhat", "prediction")
            )
        else:
            est = model_factory(model_type, model_params)
            feature_cols = [
                c
                for c in train_k.columns
                if c not in group_cols + [date_col, target_col]
            ]
            pipe = assemble_global_pipeline(
                train_k,
                target_col,
                group_cols,
                categorical_cols=group_cols,
                feature_cols=feature_cols,
                estimator=est,
            )
            model = pipe.fit(train_k)
            pred_k = model.transform(test_k).select(
                *group_cols, date_col, target_col, F.col("prediction")
            )

        by_series, portfolio = compute_metrics(
            pred_k.withColumnRenamed(target_col, "y"),
            date_col=date_col,
            actual_col="y",
            pred_col=target_col,
            group_cols=group_cols,
            mase_seasonality=mase_seasonality,
        )
        results.append(portfolio.withColumn("fold", F.lit(k)))

    return results[0].unionByName(results[1]) if len(results) > 1 else results[0]


# ----------------------------
# 8) Orchestration: train → forecast → evaluate
# ----------------------------
def run_forecast(df: DataFrame, cfg: Dict[str, Any]) -> Dict[str, DataFrame]:
    d = cfg["data"]
    s = cfg["split"]
    m = cfg["model"]
    f = cfg["features"]
    e = cfg["evaluation"]
    # Build features on full DF (then split) to keep consistent warmup

    df_feat = build_features(
        df,
        d["date_col"],
        d["target_col"],
        d["group_cols"],
        f["lags"],
        f["mas"],
        f["add_time_signals"],
    )
    train, test = train_test_split(
        df_feat,
        d["date_col"],
        d["group_cols"],
        s["mode"],
        s.get("train_end_date", ""),
        s.get("test_horizon", 28),
        d.get("min_train_periods", 56),
    )

    ic()
    ic(df_feat.show(2))
    ic(train.show(2))
    ic(test.show(2))

    if m["type"].endswith("_local"):
        udf = forecast_local_udf(
            m["type"],
            s.get("test_horizon", 28),
            d.get("freq", "D"),
            d["date_col"],
            d["target_col"],
        )
        pred = (
            train.groupBy(*d["group_cols"])
            .apply(udf)
            .join(
                test.select(*d["group_cols"], d["date_col"], d["target_col"]),
                on=d["group_cols"] + [d["date_col"]],
                how="inner",
            )
            .withColumnRenamed("yhat", "prediction")
        )
    else:
        est = model_factory(m["type"], m["params"])
        ic(est)

        # IMPORTANT: create label on the training frame
        train = train.withColumn("label", F.col(d["target_col"]).cast("double"))

        feature_cols = [
            c
            for c in train.columns
            if c not in d["group_cols"] + [d["date_col"], d["target_col"], "label"]
        ]
        ic(feature_cols)
        ic(train.select(feature_cols).show(2))

        ic(d["target_col"])
        ic(d["group_cols"])
        ic(feature_cols)
        ic(est)

        pipe = assemble_global_pipeline(
            train,
            d["target_col"],
            d["group_cols"],
            categorical_cols=d["group_cols"],
            feature_cols=feature_cols,
            estimator=est,
        )

        ic(pipe)

        model = pipe.fit(train)
        pred = model.transform(test).select(
            *d["group_cols"], d["date_col"], d["target_col"], F.col("prediction")
        )

    # by_series, portfolio = compute_metrics(
    #     pred.withColumnRenamed(d["target_col"], "y"),
    #     "y",
    #     "prediction",
    #     *d["group_cols"],
    #     e.get("mase_seasonality", 7),
    # )

    # # Optional backtest
    # bt = None
    # if e["backtest"]["enabled"]:
    #     bt = rolling_backtest(
    #         df,
    #         d["date_col"],
    #         d["target_col"],
    #         d["group_cols"],
    #         feature_params={
    #             "lags": f["lags"],
    #             "mas": f["mas"],
    #             "add_time_signals": f["add_time_signals"],
    #             "freq": d.get("freq", "D"),
    #         },
    #         model_type=m["type"],
    #         model_params=m["params"],
    #         folds=e["backtest"]["folds"],
    #         fold_horizon=e["backtest"]["fold_horizon"],
    #         step=e["backtest"]["step"],
    #         mase_seasonality=e.get("mase_seasonality", 7),
    #     )

    return {
        "predictions": pred,
        # "metrics_by_series": by_series,
        # "metrics_portfolio": portfolio,
        # "backtest": bt,
    }
