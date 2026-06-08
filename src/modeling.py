"""
modeling.py
===========
Rolling-window XGBoost training loop that produces one-month-ahead
cross-sectional return predictions for the full backtest period.

Loop structure
--------------
For each month t from 1 to (max_month - rolling_window):
    train on months [t, t + rolling_window]   (inclusive both ends)
    predict on month  t + rolling_window + 1

The first prediction is for month (rolling_window + 2). Empirically the
panel's first month_num is June 1992, so month 62 is July 1997 -- that is
where the rolling backtest's first prediction lands. Predictions then run
monthly through the panel's final month (December 2024), giving 330 test
months in total. (The portfolio backtest in portfolio_construction.py
later filters to yr >= 2000, leaving 300 months: January 2000 onward.)

A note on window size
---------------------
The variable is named `rolling_window = 60` and the report describes a
"60-month rolling window." However, the slice
    month_num >= t AND month_num <= t + rolling_window
is INCLUSIVE on both endpoints, so the actual training window contains
61 monthly cohorts, not 60. This was the behavior in the original
notebook and was used to produce the originally published Sharpe 2.52 /
alpha 4.43% results, so we preserve it verbatim here. (Those original
figures were later found to be inflated by a vol_12m look-ahead leak; the
corrected, leak-free headline is Sharpe ~1.03 / alpha +2.19% -- see
docs/leakage_audit.md. The 61-cohort window itself is unrelated to the
leak and is kept for reproducibility.) Changing the slice to get exactly
60 months would change the numbers.

Outputs
-------
1. predicted_ret_df : long-format predictions with one row per
   (PERMNO, month) over the full test period.
2. r2_df : per-month out-of-sample R² values.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import xgboost as xgb

from feature_engineering import FEATURE_PCT_RANK_COLS

logger = logging.getLogger(__name__)


# Columns retained in the final predictions DataFrame
PREDICTION_OUTPUT_COLS = [
    "date", "PERMNO", "yr", "month",
    "RET", "adj_ret", "predicted_adj_ret",
]


def train_one_window(
    train_dat: pd.DataFrame,
    test_dat: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_params: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    """Fit XGBoost on a single training window and predict the next month.

    Parameters
    ----------
    train_dat
        Rows belonging to the rolling training window.
    test_dat
        Rows belonging to the single test month immediately after the
        training window.
    feature_cols
        Column names to use as model inputs (the 10 _pct_rank columns).
    target_col
        Column name of the regression target (typically 'adj_ret').
    model_params
        Keyword arguments for xgb.XGBRegressor.

    Returns
    -------
    test_dat_with_pred : DataFrame
        Copy of test_dat with a new `predicted_adj_ret` column.
    r2 : float
        Out-of-sample R² for this test month.

    Notes
    -----
    Features and targets are cast to float32 for memory efficiency and
    to match XGBoost's preferred input dtype. The XGBoost `tree_method`
    ('hist' by default in our config) handles float32 natively without
    precision loss for this kind of cross-sectional ranking problem.
    """
    X_train = train_dat[feature_cols].astype("float32")
    y_train = train_dat[target_col].astype("float32")
    X_test = test_dat[feature_cols].astype("float32")
    y_test = test_dat[target_col].astype("float32")

    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train, verbose=False)

    test_dat = test_dat.copy()
    test_dat["predicted_adj_ret"] = model.predict(X_test)
    r2 = model.score(X_test, y_test)

    return test_dat, r2


def run_rolling_backtest(
    dat: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    rolling_window_months: int,
    model_params: dict[str, Any],
    log_every_n_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full rolling-window XGBoost backtest.

    For each month t in [1, max_month - rolling_window):
        - Train on months [t, t + rolling_window] (inclusive both ends)
        - Predict the next month (t + rolling_window + 1)

    Parameters
    ----------
    dat
        Feature-engineered modeling panel from
        feature_engineering.run_feature_engineering(). Must have
        a `month_num` column and all `feature_cols`.
    feature_cols
        Column names of the 10 percentile-rank features.
    target_col
        Name of the demeaned return target ('adj_ret').
    rolling_window_months
        Configured window size. NOTE: the actual slice contains
        (rolling_window_months + 1) months due to the inclusive-both-ends
        slice. See module docstring.
    model_params
        XGBoost hyperparameters dictionary.
    log_every_n_months
        Print a progress log line every N iterations. 12 = once per
        simulated year, which is a useful cadence for a 330-month run.

    Returns
    -------
    predicted_ret_df : DataFrame
        Long-format predictions, one row per (PERMNO, month) across
        all test months. Columns: date, PERMNO, yr, month, RET,
        adj_ret, predicted_adj_ret.
    r2_df : DataFrame
        Per-month diagnostic table with columns: date, r2, n_test.
    """
    max_month = int(dat["month_num"].max())
    start_month_limit = max_month - rolling_window_months
    n_iterations = start_month_limit - 1

    logger.info(
        "Starting rolling backtest: %d iterations, window=%d months (inclusive both ends), "
        "max_month=%d, n_features=%d",
        n_iterations, rolling_window_months, max_month, len(feature_cols),
    )

    pred_parts: list[pd.DataFrame] = []
    r2_rows: list[dict[str, Any]] = []

    for t in range(1, start_month_limit):
        train_month_end = t + rolling_window_months
        test_month = train_month_end + 1

        train_dat = dat[
            (dat["month_num"] >= t) & (dat["month_num"] <= train_month_end)
        ]
        test_dat = dat[dat["month_num"] == test_month].copy()

        if len(test_dat) == 0:
            # Defensive: skip if no test rows. The original notebook
            # didn't guard against this; in practice it never happens
            # because every month in the panel has stocks.
            continue

        test_dat_with_pred, r2 = train_one_window(
            train_dat=train_dat,
            test_dat=test_dat,
            feature_cols=feature_cols,
            target_col=target_col,
            model_params=model_params,
        )

        r2_rows.append({
            "date": test_dat_with_pred["date"].iloc[0],
            "r2": r2,
            "n_test": len(test_dat_with_pred),
        })
        pred_parts.append(test_dat_with_pred[PREDICTION_OUTPUT_COLS])

        # Periodic progress log
        if t % log_every_n_months == 0 or t == 1:
            logger.info(
                "  iter %d/%d  train=[%d, %d]  test=%d  r2=%.4f  n_test=%d",
                t, n_iterations, t, train_month_end, test_month,
                r2, len(test_dat_with_pred),
            )

    predicted_ret_df = pd.concat(pred_parts, ignore_index=True)
    r2_df = pd.DataFrame(r2_rows)

    logger.info(
        "Backtest complete: %s predictions across %d months. Avg R²=%.4f, median R²=%.4f",
        f"{len(predicted_ret_df):,}", len(r2_df),
        r2_df["r2"].mean(), r2_df["r2"].median(),
    )

    return predicted_ret_df, r2_df


def run_modeling(
    dat: pd.DataFrame,
    rolling_window_months: int,
    model_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the modeling pipeline end-to-end.

    Thin wrapper around run_rolling_backtest() that uses the module
    constants FEATURE_PCT_RANK_COLS and the 'adj_ret' target. This is
    what run_pipeline.py calls.

    Returns
    -------
    predicted_ret_df, r2_df : two DataFrames as documented above.
    """
    return run_rolling_backtest(
        dat=dat,
        feature_cols=FEATURE_PCT_RANK_COLS,
        target_col="adj_ret",
        rolling_window_months=rolling_window_months,
        model_params=model_params,
    )
