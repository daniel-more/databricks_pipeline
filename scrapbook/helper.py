# smartforecast/forecasting.py
from typing import Any, Dict, List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


# ---------------------------------------------------------------------
# Aggregation and validation
# ---------------------------------------------------------------------
def aggregate_to_granularity(
    df: DataFrame,
    date_col: str,
    target_col: str,
    group_cols: List[str],
    agg: str = "sum",
    extra_numeric_aggs: Dict[str, str] = None,
) -> DataFrame:
    """
    Aggregates to exactly one row per (group_cols, date_col).
    - agg: aggregation for the target ('sum'|'mean'|'avg'|'max'|'min')
    - extra_numeric_aggs: optional dict {col_name: agg} for exogenous numeric cols.
    """
    funs = {"sum": F.sum, "mean": F.avg, "avg": F.avg, "max": F.max, "min": F.min}
    if agg.lower() not in funs:
        raise ValueError(f"Unsupported agg='{agg}'")
    metrics = [funs[agg.lower()](F.col(target_col)).alias(target_col)]
    if extra_numeric_aggs:
        for c, a in extra_numeric_aggs.items():
            if a.lower() not in funs:
                raise ValueError(f"Unsupported agg '{a}' for '{c}'")
            metrics.append(funs[a.lower()](F.col(c)).alias(c))
    gdf = (
        df.groupBy(*group_cols, date_col)
        .agg(*metrics)
        .withColumn(date_col, F.to_date(F.col(date_col)))
    )
    return gdf


def assert_unique_series_rows(df: DataFrame, date_col: str, group_cols: List[str]):
    """Ensure one row per (group,date)."""
    dup = (
        df.groupBy(*group_cols, date_col)
        .agg(F.count("*").alias("cnt"))
        .filter(F.col("cnt") > 1)
    )
    if dup.head(1):
        raise AssertionError("Duplicates per (group,date). Pre-aggregate first.")


# ---------------------------------------------------------------------
# Feature engineering (correct: no window-in-agg usage)
# ---------------------------------------------------------------------
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
    Creates lag and moving average features.
    - Pre-aggregates to selected granularity if requested.
    - Adds calendar signals and drops warm-up rows.
    """
    if pre_aggregate:
        df = aggregate_to_granularity(
            df,
            date_col,
            target_col,
            group_cols,
            agg=target_agg,
            extra_numeric_aggs=extra_numeric_aggs,
        )
        assert_unique_series_rows(df, date_col, group_cols)

    w = Window.partitionBy(*group_cols).orderBy(F.col(date_col))

    # lags
    for L in lags or []:
        df = df.withColumn(f"lag_{L}", F.lag(F.col(target_col), L).over(w))

    # moving averages (rolling window)
    for M in mas or []:
        df = df.withColumn(
            f"ma_{M}", F.avg(F.col(target_col)).over(w.rowsBetween(-M + 1, 0))
        )

    # calendar/time signals
    if add_time_signals:
        df = (
            df.withColumn("dow", F.dayofweek(F.col(date_col)))
            .withColumn("dom", F.dayofmonth(F.col(date_col)))
            .withColumn("weekofyear", F.weekofyear(F.col(date_col)))
            .withColumn("month", F.month(F.col(date_col)))
            .withColumn("year", F.year(F.col(date_col)))
        )

    # warm-up drop
    warmup = max([0] + (lags or []) + (mas or []))
    if warmup > 0:
        df = (
            df.withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") > warmup)
            .drop("_rn")
        )

    return df


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------
def train_test_split(
    df: DataFrame,
    date_col: str,
    group_cols: List[str],
    mode: str = "horizon",
    train_end_date: str = "",
    test_horizon: int = 28,
    min_train_periods: int = 56,
) -> Tuple[DataFrame, DataFrame]:
    """Date-based or horizon-based split per series."""
    counts = df.groupBy(*group_cols).agg(F.count("*").alias("n"))
    valid = counts.filter(
        F.col("n") >= (min_train_periods + (test_horizon if mode == "horizon" else 0))
    ).select(*group_cols)
    df = df.join(valid, on=group_cols, how="inner")

    if mode == "date" and train_end_date:
        train = df.filter(F.col(date_col) <= F.to_date(F.lit(train_end_date)))
        test = df.filter(F.col(date_col) > F.to_date(F.lit(train_end_date)))
        return train, test

    w = Window.partitionBy(*group_cols).orderBy(F.col(date_col))
    with_idx = df.withColumn("rnk", F.row_number().over(w))
    max_idx = with_idx.groupBy(*group_cols).agg(F.max("rnk").alias("max_rnk"))
    joined = with_idx.join(max_idx, on=group_cols)

    test = joined.filter(F.col("rnk") > (F.col("max_rnk") - F.lit(test_horizon))).drop(
        "rnk", "max_rnk"
    )
    train = joined.filter(
        F.col("rnk") <= (F.col("max_rnk") - F.lit(test_horizon))
    ).drop("rnk", "max_rnk")
    return train, test


# ---------------------------------------------------------------------
# Model factory and pipeline (global models)
# ---------------------------------------------------------------------
def model_factory(model_type: str, params: Dict[str, Any]):
    if model_type == "spark_gbt":
        return GBTRegressor(featuresCol="features", labelCol="label", **params)
    if model_type == "spark_rf":
        return RandomForestRegressor(featuresCol="features", labelCol="label", **params)
    if model_type == "glm":
        return LinearRegression(featuresCol="features", labelCol="label", **params)
    if model_type in {"prophet_local", "arima_local"}:
        return model_type  # handled separately
    raise ValueError(f"Unknown model_type: {model_type}")


def assemble_global_pipeline(
    target_col: str,
    group_cols: List[str],
    categorical_cols: List[str],
    feature_cols: List[str],
    estimator,
) -> Pipeline:
    stages = []
    encoded = []
    for c in categorical_cols:
        si = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_ohe"])
        stages += [si, ohe]
        encoded.append(f"{c}_ohe")
    assembler = VectorAssembler(inputCols=feature_cols + encoded, outputCol="features")
    stages += [assembler, estimator]
    return Pipeline(stages=stages)


def fit_global_model(
    train: DataFrame,
    target_col: str,
    group_cols: List[str],
    feature_cols: List[str],
    estimator,
):
    """Add label, build pipeline, fit global model."""
    train = train.withColumn("label", F.col(target_col).cast("double"))
    pipe = assemble_global_pipeline(
        target_col,
        group_cols,
        categorical_cols=group_cols,
        feature_cols=feature_cols,
        estimator=estimator,
    )
    model = pipe.fit(train)
    return model


def predict_global(
    model, test: DataFrame, group_cols: List[str], date_col: str, target_col: str
) -> DataFrame:
    """Predict using fitted global model."""
    return model.transform(test).select(
        *group_cols, date_col, F.col(target_col).alias("y"), F.col("prediction")
    )


# ---------------------------------------------------------------------
# Metrics (correct: materialize window columns first)
# ---------------------------------------------------------------------
def compute_metrics(
    pred_df: DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    group_cols: List[str],
    mase_seasonality: int = 7,
    epsilon: float = 1e-12,
) -> Tuple[DataFrame, DataFrame]:
    """
    sMAPE, wMAPE, MASE at per-series and portfolio level.
    Avoids window-in-agg by creating 'naive' lag and row-wise errors first.
    """
    df = pred_df.withColumn(actual_col, F.col(actual_col).cast("double")).withColumn(
        pred_col, F.col(pred_col).cast("double")
    )

    w = Window.partitionBy(*group_cols).orderBy(F.col(date_col))
    df = df.withColumn("naive", F.lag(F.col(actual_col), mase_seasonality).over(w))

    df = (
        df.withColumn(
            "smape_row",
            F.when(
                (F.abs(F.col(actual_col)) + F.abs(F.col(pred_col))) > 0,
                F.abs(F.col(actual_col) - F.col(pred_col))
                / ((F.abs(F.col(actual_col)) + F.abs(F.col(pred_col))) / 2.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("abs_err", F.abs(F.col(actual_col) - F.col(pred_col)))
        .withColumn("mae_forecast_row", F.abs(F.col(actual_col) - F.col(pred_col)))
        .withColumn("mae_naive_row", F.abs(F.col(actual_col) - F.col("naive")))
    )

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


# ---------------------------------------------------------------------
# Local models via top-level pandas UDFs (keeps task binary small)
# ---------------------------------------------------------------------
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DateType, DoubleType, StructField, StructType

LOCAL_PARAMS: Dict[str, Any] = {}
_schema_local = StructType(
    [StructField("date", DateType(), False), StructField("yhat", DoubleType(), False)]
)


@pandas_udf(_schema_local, F.PandasUDFType.GROUPED_MAP)
def prophet_udf(pdf):
    """Per-group Prophet forecast; parameters pulled from LOCAL_PARAMS."""
    import pandas as pd
    from prophet import Prophet

    date_col = LOCAL_PARAMS["date_col"]
    target_col = LOCAL_PARAMS["target_col"]
    horizon = LOCAL_PARAMS["horizon"]
    freq = LOCAL_PARAMS["freq"]
    df_fit = pdf[[date_col, target_col]].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    m = Prophet()
    m.fit(df_fit)
    future = m.make_future_dataframe(periods=horizon, freq=freq)
    yhat = (
        m.predict(future).tail(horizon)[["ds", "yhat"]].rename(columns={"ds": "date"})
    )
    return yhat


@pandas_udf(_schema_local, F.PandasUDFType.GROUPED_MAP)
def arima_udf(pdf):
    """Per-group ARIMA forecast; parameters pulled from LOCAL_PARAMS."""
    import pandas as pd
    import statsmodels.api as sm

    date_col = LOCAL_PARAMS["date_col"]
    target_col = LOCAL_PARAMS["target_col"]
    horizon = LOCAL_PARAMS["horizon"]
    freq = LOCAL_PARAMS["freq"]
    order = LOCAL_PARAMS.get("order", (1, 1, 1))
    y = pdf[target_col].astype(float)
    model = sm.tsa.ARIMA(y, order=order).fit()
    fc = model.forecast(steps=horizon)
    dates = pd.date_range(pdf[date_col].max(), periods=horizon + 1, freq=freq)[1:]
    return pd.DataFrame({"date": dates, "yhat": fc})


def fit_predict_local(
    train: DataFrame,
    test: DataFrame,
    model_type: str,
    date_col: str,
    target_col: str,
    group_cols: List[str],
    horizon: int,
    freq: str = "D",
    arima_order: Tuple[int, int, int] = (1, 1, 1),
) -> DataFrame:
    """Train local model per group and forecast horizon steps; join with test actuals."""
    LOCAL_PARAMS.update(
        {
            "date_col": date_col,
            "target_col": target_col,
            "horizon": horizon,
            "freq": freq,
            "order": arima_order,
        }
    )
    udf = prophet_udf if model_type == "prophet_local" else arima_udf
    pred = (
        train.groupBy(*group_cols)
        .apply(udf)
        .join(
            test.select(*group_cols, date_col, target_col),
            on=[date_col] + group_cols,
            how="inner",
        )
        .withColumnRenamed("yhat", "prediction")
        .withColumnRenamed(target_col, "y")
    )
    return pred


# ---------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------
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
    """Rolling-origin evaluation; returns portfolio metrics per fold."""
    results = []
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

        # Build features on already-aggregated slices (no further pre_aggregate)
        train_k = build_features(
            train_k,
            date_col,
            target_col,
            group_cols,
            lags=feature_params["lags"],
            mas=feature_params["mas"],
            add_time_signals=feature_params["add_time_signals"],
            pre_aggregate=False,
        )
        test_k = build_features(
            test_k,
            date_col,
            target_col,
            group_cols,
            lags=feature_params["lags"],
            mas=feature_params["mas"],
            add_time_signals=feature_params["add_time_signals"],
            pre_aggregate=False,
        )

        if model_type.endswith("_local"):
            pred_k = fit_predict_local(
                train_k,
                test_k,
                model_type,
                date_col,
                target_col,
                group_cols,
                horizon=fold_horizon,
                freq=feature_params.get("freq", "D"),
                arima_order=model_params.get("order", (1, 1, 1)),
            )
        else:
            est = model_factory(model_type, model_params)
            feature_cols = [
                c
                for c in train_k.columns
                if c not in group_cols + [date_col, target_col, "label"]
            ]
            model = fit_global_model(train_k, target_col, group_cols, feature_cols, est)
            pred_k = predict_global(model, test_k, group_cols, date_col, target_col)

        _, portfolio = compute_metrics(
            pred_k, date_col, "y", "prediction", group_cols, mase_seasonality
        )
        results.append(portfolio.withColumn("fold", F.lit(k)))

    out = results[0]
    for i in range(1, len(results)):
        out = out.unionByName(results[i])
    return out


# ---------------------------------------------------------------------
# Orchestration (optional): single call to run the pipeline
# ---------------------------------------------------------------------
def run_forecast(df: DataFrame, cfg: Dict[str, Any]) -> Dict[str, DataFrame]:
    d, a, f, s, m, e = (
        cfg["data"],
        cfg.get("aggregation", {}),
        cfg["features"],
        cfg["split"],
        cfg["model"],
        cfg["evaluation"],
    )

    # 1) aggregate & features
    df_feat = build_features(
        df,
        d["date_col"],
        d["target_col"],
        d["group_cols"],
        lags=f["lags"],
        mas=f["mas"],
        add_time_signals=f["add_time_signals"],
        pre_aggregate=True,
        target_agg=a.get("target_agg", "sum"),
        extra_numeric_aggs=a.get("extra_numeric_aggs"),
    )

    # 2) split
    train, test = train_test_split(
        df_feat,
        d["date_col"],
        d["group_cols"],
        s["mode"],
        s.get("train_end_date", ""),
        s["test_horizon"],
        d["min_train_periods"],
    )

    # 3) train & predict
    if m["type"].endswith("_local"):
        pred = fit_predict_local(
            train,
            test,
            m["type"],
            d["date_col"],
            d["target_col"],
            d["group_cols"],
            horizon=s["test_horizon"],
            freq=d.get("freq", "D"),
            arima_order=m.get("params", {}).get("order", (1, 1, 1)),
        )
    else:
        est = model_factory(m["type"], m["params"])
        feature_cols = [
            c
            for c in train.columns
            if c not in d["group_cols"] + [d["date_col"], d["target_col"], "label"]
        ]
        model = fit_global_model(
            train, d["target_col"], d["group_cols"], feature_cols, est
        )
        pred = predict_global(
            model, test, d["group_cols"], d["date_col"], d["target_col"]
        )

    # 4) metrics
    by_series, portfolio = compute_metrics(
        pred, d["date_col"], "y", "prediction", d["group_cols"], e["mase_seasonality"]
    )

    # 5) optional backtest
    bt = None
    if e["backtest"]["enabled"]:
        df_agg = aggregate_to_granularity(
            df,
            d["date_col"],
            d["target_col"],
            d["group_cols"],
            agg=a.get("target_agg", "sum"),
            extra_numeric_aggs=a.get("extra_numeric_aggs"),
        )
        bt = rolling_backtest(
            df_agg,
            d["date_col"],
            d["target_col"],
            d["group_cols"],
            feature_params={
                "lags": f["lags"],
                "mas": f["mas"],
                "add_time_signals": f["add_time_signals"],
                "freq": d.get("freq", "D"),
            },
            model_type=m["type"],
            model_params=m["params"],
            folds=e["backtest"]["folds"],
            fold_horizon=e["backtest"]["fold_horizon"],
            step=e["backtest"]["step"],
            mase_seasonality=e["mase_seasonality"],
        )

    return {
        "features": df_feat,
        "train": train,
        "test": test,
        "predictions": pred,
        "metrics_by_series": by_series,
        "metrics_portfolio": portfolio,
        "backtest": bt,
    }

    return {
        "features": df_feat,
        "train": train,
        "test": test,
        "predictions": pred,
        "metrics_by_series": by_series,
        "metrics_portfolio": portfolio,
        "backtest": bt,
    }

    return {
        "features": df_feat,
        "train": train,
        "test": test,
        "predictions": pred,
        "metrics_by_series": by_series,
        "metrics_portfolio": portfolio,
        "backtest": bt,
    }


from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, Window


def plot_forecast(
    pred_df: DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    group_cols: List[str],
    series_id: Dict[str, str] = None,
    title: str = "Forecast vs Actual",
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot actual vs predicted for a single series.
    Parameters
    ----------
    pred_df : DataFrame
        Must include date_col, actual_col, pred_col, and group_cols.
    series_id : dict
        Filter for one series, e.g., {"family": "BEVERAGES"} or {"family": "BEVERAGES", "store_nbr": "1"}.
    """
    # Filter for one series
    # After filtering and sorting

    pdf = pred_df.sort_values(date_col)
    print(pdf.head(5))

    # Aggregate only if needed
    if pdf.duplicated(subset=[date_col]).any():
        pdf = (
            pdf.groupby([date_col] + group_cols, as_index=False)
            .agg({actual_col: "sum", pred_col: "sum"})
            .sort_values(date_col)
        )

    if series_id:
        for k, v in series_id.items():
            pdf = pdf[pdf[k] == v]

    pdf = pdf.sort_values(date_col)

    plt.figure(figsize=figsize)
    plt.plot(pdf[date_col], pdf[actual_col], label="Actual", color="blue")
    plt.plot(
        pdf[date_col], pdf[pred_col], label="Prediction", color="orange", linestyle="--"
    )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(actual_col)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_train_test_forecast(
    train_df,
    test_df,
    pred_df,
    date_col,
    target_col,
    pred_col,
    group_cols,
    series_id,
    last_n_train: int = None,  # NEW: number of last train points to show
    figsize=(10, 5),
):
    # Convert to Pandas
    train_pdf = train_df.toPandas()
    test_pdf = test_df.toPandas()
    pred_pdf = pred_df.toPandas()

    # Filter for one series
    for k, v in series_id.items():
        train_pdf = train_pdf[train_pdf[k] == v]
        test_pdf = test_pdf[test_pdf[k] == v]
        pred_pdf = pred_pdf[pred_pdf[k] == v]

    # Sort by date
    train_pdf = train_pdf.sort_values(date_col)
    test_pdf = test_pdf.sort_values(date_col)
    pred_pdf = pred_pdf.sort_values(date_col)

    # Apply last_n_train filter if provided
    if last_n_train is not None and last_n_train > 0:
        train_pdf = train_pdf.tail(last_n_train)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.plot(
        train_pdf[date_col], train_pdf[target_col], label="Train Actual", color="blue"
    )
    plt.plot(
        test_pdf[date_col], test_pdf[target_col], label="Test Actual", color="green"
    )
    plt.plot(
        pred_pdf[date_col],
        pred_pdf[pred_col],
        label="Prediction",
        color="orange",
        linestyle="--",
    )
    plt.title(f"Forecast for {series_id}")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.show()
