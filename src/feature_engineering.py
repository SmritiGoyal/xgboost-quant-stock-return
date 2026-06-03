"""
feature_engineering.py
======================
Construct the 10 cross-sectional features used by the XGBoost model,
plus the regression target (cross-sectionally demeaned return).

Pipeline stages:
    1. Compute 10 features from the merged CRSP/Compustat panel.
    2. Cross-sectional Winsorization at 1%/99% (per-month, per-feature).
    3. Cross-sectional percentile rank transformation (per-month).
    4. Add date utilities (yr, month, month_num) used by the rolling
       training loop.
    5. Compute the cross-sectionally demeaned return (adj_ret) as the
       regression target.

Why 10 features and not 9
-------------------------
The final report describes 9 features (7 academic + 2 new). The notebook
code, however, passes 10 features into XGBoost: the 9 documented features
PLUS `marketcap` itself, used as an implicit size control.

We preserve the notebook behavior here (10 features) because that is what
the published Sharpe 2.52 / alpha 4.43% results were produced from. The
README and methodology docs call this discrepancy out explicitly.

Feature list (in the exact order they appear in the model input matrix):
    1. marketcap      — lagged market capitalization ($K)
    2. new_issue      — 12-month split-adjusted share growth, lagged 1 mo
    3. investment     — year-over-year total asset growth
    4. accruals       — (income before extraordinary - operating cash flow) / total assets
    5. b2m            — book equity / market cap (value factor)
    6. ret_2_12       — cumulative return from t-12 to t-2 (intermediate momentum)
    7. CashFlow2TA    — operating cash flow / total assets
    8. CashFlow2Prc   — operating cash flow / market cap
    9. reversal_1m    — negative of last month's return (short-term reversal)
   10. vol_12m        — rolling 12-month return standard deviation

Academic citations
------------------
- ret_2_12     : Jegadeesh & Titman (1993) — Returns to buying winners and
                 selling losers
- new_issue    : Daniel, Hirshleifer, Sun (2020) — Short- and long-horizon
                 behavioral factors
- investment   : Fama & French (2015) — A five-factor asset pricing model
- accruals     : Sloan (1996) — Earnings quality / accruals anomaly
- b2m          : Fama & French (1992) — value factor
- reversal_1m  : Jegadeesh (1990), Lehmann (1990) — short-term reversal
- vol_12m      : Ang, Hodrick, Xing, Zhang (2006) — idiosyncratic volatility puzzle
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# FEATURE NAMES — order matters for model input matrix
# =====================================================================

FEATURES = [
    "marketcap",
    "new_issue",
    "investment",
    "accruals",
    "b2m",
    "ret_2_12",
    "CashFlow2TA",
    "CashFlow2Prc",
    "reversal_1m",
    "vol_12m",
]

# Columns produced by add_percentile_ranks() — these are the actual
# inputs to XGBoost (the raw features are kept for diagnostics).
FEATURE_PCT_RANK_COLS = [f"{f}_pct_rank" for f in FEATURES]

# Momentum lag range: cumulative return from t-12 to t-2 inclusive
# (11 monthly lags, skipping the most recent month to avoid microstructure
# noise from short-term reversal — that signal is captured separately in
# reversal_1m).
MOMENTUM_LAG_START = 2
MOMENTUM_LAG_END = 13   # range() upper-exclusive, so this gives 2..12


def compute_raw_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Add the 10 features as new columns on the merged panel.

    Parameters
    ----------
    merged
        Output of ingestion.run_ingestion() — must contain columns:
        PERMNO, date, RET, SHROUT_adj, marketcap, at, lag_at, ib, oancf,
        ceq.

    Returns
    -------
    DataFrame with the 10 feature columns added. Intermediate columns
    (SHROUT_adj_lag12, new_issue_asof_monthend) are also retained for
    diagnostic transparency but are not used downstream.

    Notes
    -----
    The `new_issue` feature applies an additional `.shift(1)` after the
    12-month change, meaning the published feature value at month t
    reflects the change between t-13 and t-1 — a one-month conservative
    buffer that ensures the share count is fully reported. This is the
    convention used in the original notebook.

    The momentum feature `ret_2_12` deliberately skips month t-1 to avoid
    contamination from the short-term reversal effect, which is captured
    separately by `reversal_1m`.
    """
    logger.info("Computing raw features on %s merged rows", f"{len(merged):,}")

    g = merged.groupby("PERMNO")

    # -----------------------------------------------------------------
    # new_issue — 12-month split-adjusted share growth, lagged 1 month
    # -----------------------------------------------------------------
    merged["SHROUT_adj_lag12"] = g["SHROUT_adj"].shift(12)
    merged["new_issue_asof_monthend"] = (
        (merged["SHROUT_adj"] - merged["SHROUT_adj_lag12"])
        / merged["SHROUT_adj_lag12"]
    )
    # Extra .shift(1) ensures the value used at month t was definitively
    # publicly available — the published value reflects t-13 to t-1.
    merged["new_issue"] = g["new_issue_asof_monthend"].shift(1)

    # -----------------------------------------------------------------
    # ret_2_12 — cumulative return from t-12 to t-2 (skip t-1)
    # -----------------------------------------------------------------
    # mom_t = (1+RET_{t-2}) * (1+RET_{t-3}) * ... * (1+RET_{t-12}) - 1
    # This is the standard "intermediate-horizon momentum" formulation
    # following Jegadeesh & Titman (1993).
    mom = pd.Series(1.0, index=merged.index)
    for i in range(MOMENTUM_LAG_START, MOMENTUM_LAG_END):
        mom = mom * (1 + g["RET"].shift(i))
    merged["ret_2_12"] = mom - 1

    # -----------------------------------------------------------------
    # Compustat-derived features
    # -----------------------------------------------------------------
    merged["investment"] = merged["at"] / merged["lag_at"] - 1
    merged["accruals"] = (merged["ib"] - merged["oancf"]) / merged["at"]
    merged["b2m"] = merged["ceq"] / merged["marketcap"]
    merged["CashFlow2TA"] = merged["oancf"] / merged["at"]
    merged["CashFlow2Prc"] = merged["oancf"] / merged["marketcap"]

    # -----------------------------------------------------------------
    # Short-horizon CRSP-only features
    # -----------------------------------------------------------------
    # reversal_1m: negate last month's return so positive values predict
    # higher subsequent returns (consistent with the rank-then-decile sort).
    merged["reversal_1m"] = -g["RET"].shift(1)

    # vol_12m: trailing 12-month standard deviation of returns.
    # The .reset_index(level=0, drop=True) is required because
    # groupby+rolling returns a MultiIndex (PERMNO, original_index)
    # which doesn't align with merged's flat index for assignment.
    #
    # LEAKAGE FIX: rolling(12) is right-aligned, so at month t the window
    # is [t-11, t] -- it INCLUDES month t's own return. Since the model
    # target is month t's (demeaned) return, an unlagged value lets the
    # feature peek at the very return being predicted. We shift by one
    # month so the window ends at t-1 (information available at the start
    # of the prediction month), matching the lag discipline applied to
    # every other feature (reversal_1m, new_issue, ret_2_12, etc.).
    #
    # The unlagged version inflates the long-short Sharpe to ~2.5; the
    # lagged (correct) version yields the honest ~1.03. See
    # docs/leakage_audit.md for the full ablation.
    merged["vol_12m"] = (
        g["RET"].rolling(12).std().reset_index(level=0, drop=True)
    )
    merged["vol_12m"] = merged.groupby("PERMNO")["vol_12m"].shift(1)

    logger.info("  Raw features computed: %s", FEATURES)
    return merged


def select_modeling_panel(merged: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing feature and select modeling columns.

    Parameters
    ----------
    merged
        Output of compute_raw_features().

    Returns
    -------
    DataFrame with columns [date, PERMNO, RET] + FEATURES, with all
    rows having complete feature values.
    """
    cols = ["date", "PERMNO", "RET"] + FEATURES
    dat = merged[cols].copy().dropna().reset_index(drop=True)
    logger.info("  Modeling panel: %s rows after dropna", f"{len(dat):,}")
    return dat


def winsorize_series(x: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    """Clip a series to the [low, high] percentile range.

    Parameters
    ----------
    x
        Numeric series to winsorize.
    low, high
        Lower and upper quantile bounds (in [0, 1]).

    Returns
    -------
    Series clipped to [x.quantile(low), x.quantile(high)] inclusive.
    Length-preserving — no observations are dropped.
    """
    lo = x.quantile(low)
    hi = x.quantile(high)
    return x.clip(lo, hi)


def apply_cross_sectional_winsorization(
    dat: pd.DataFrame,
    low: float,
    high: float,
) -> pd.DataFrame:
    """Winsorize each feature (and RET) per-month cross-sectionally.

    Parameters
    ----------
    dat
        Output of select_modeling_panel().
    low, high
        Quantile bounds (typically 0.01 and 0.99).

    Returns
    -------
    Same DataFrame with features and RET clipped to the cross-sectional
    [low, high] percentile bounds *within each month*.

    Notes
    -----
    Per-month clipping is essential because feature distributions shift
    over time. A naive pooled Winsorize would clip values from the early
    1990s using bounds dominated by the late 2010s, distorting the
    relative ordering within any single month.
    """
    cols_to_winsorize = FEATURES + ["RET"]
    for col in cols_to_winsorize:
        dat[col] = dat.groupby("date")[col].transform(
            lambda x, lo=low, hi=high: winsorize_series(x, lo, hi)
        )
    logger.info(
        "  Winsorized %d cols at [%.2f, %.2f] per-month",
        len(cols_to_winsorize), low, high,
    )
    return dat


def add_percentile_ranks(dat: pd.DataFrame) -> pd.DataFrame:
    """Convert each feature to its cross-sectional percentile rank.

    For each feature column, adds a `_pct_rank` column containing the
    rank of that value within its month, scaled to [0, 1].

    Parameters
    ----------
    dat
        DataFrame containing the raw features (after Winsorization).

    Returns
    -------
    Same DataFrame with new `<feature>_pct_rank` columns appended.

    Notes
    -----
    Rank transformation is a leakage-safe scaling strategy because the
    rank of a value depends only on the cross-sectional ordering at that
    point in time, not on any future-information statistics (like a
    pooled std dev). It also makes the model insensitive to any
    residual outliers that survived Winsorization.
    """
    for sig in FEATURES:
        dat[f"{sig}_pct_rank"] = dat.groupby("date")[sig].rank(pct=True)
    logger.info("  Added %d percentile-rank columns", len(FEATURES))
    return dat


def add_date_utilities(dat: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, and sequential month_num columns.

    Parameters
    ----------
    dat
        DataFrame with a `date` column.

    Returns
    -------
    Same DataFrame with three new columns:
        yr        — calendar year (int)
        month     — calendar month (int 1..12)
        month_num — sequential month index (1, 2, 3, ...) using
                    pd.rank(method='dense'), so consecutive unique
                    dates get consecutive integers.

    Notes
    -----
    `month_num` is used by the rolling training loop in modeling.py to
    select 60-month training windows. Using dense ranking ensures that
    even if some months have zero rows (rare in this panel), the index
    increments cleanly.
    """
    dat["yr"] = dat["date"].dt.year
    dat["month"] = dat["date"].dt.month
    dat = dat.sort_values(["date", "PERMNO"]).reset_index(drop=True)
    dat["month_num"] = dat["date"].rank(method="dense").astype(int)
    logger.info(
        "  Date range: %s to %s (%d unique months)",
        dat["date"].min(), dat["date"].max(), dat["month_num"].max(),
    )
    return dat


def add_demeaned_target(dat: pd.DataFrame) -> pd.DataFrame:
    """Add the cross-sectionally demeaned return as the regression target.

    Parameters
    ----------
    dat
        DataFrame with RET and date columns.

    Returns
    -------
    Same DataFrame with two new columns:
        mean_ret — cross-sectional mean of RET within each month
        adj_ret  — RET minus mean_ret (the modeling target)

    Notes
    -----
    Demeaning the target by month is the standard cross-sectional
    factor-model setup: we're predicting the *relative* return ranking
    within each month, not the absolute return. This removes the
    market-level return component (which is captured separately by the
    CAPM regression in market_model.py).
    """
    dat["mean_ret"] = dat["date"].map(dat.groupby("date")["RET"].mean())
    dat["adj_ret"] = dat["RET"] - dat["mean_ret"]
    return dat


def filter_to_coverage_start_year(dat: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Filter to rows from the coverage start year onward.

    The notebook filters with `yr >= 1995` after computing features
    because the 12-month-lag features require 12 prior months of data
    to be valid. The 1995 cutoff is a conservative buffer.

    Parameters
    ----------
    dat
        DataFrame with yr column.
    start_year
        Minimum calendar year to keep (inclusive).

    Returns
    -------
    Filtered DataFrame.
    """
    n_before = len(dat)
    dat = dat[dat["yr"] >= start_year].copy()
    logger.info(
        "  Filtered to yr >= %d: %s -> %s rows",
        start_year, f"{n_before:,}", f"{len(dat):,}",
    )
    return dat


def run_feature_engineering(
    merged: pd.DataFrame,
    winsorize_low: float,
    winsorize_high: float,
    data_start_year: int,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline end-to-end.

    Combines all stages: raw feature computation, modeling panel
    selection, cross-sectional Winsorization, percentile ranks, date
    utilities, target construction, and the coverage start year filter.

    This is what run_pipeline.py calls.

    Parameters
    ----------
    merged
        Output of ingestion.run_ingestion().
    winsorize_low, winsorize_high
        Cross-sectional Winsorization bounds.
    data_start_year
        Year to filter from (inclusive).

    Returns
    -------
    Modeling-ready DataFrame with columns:
        date, PERMNO, RET, <10 raw features>,
        <10 _pct_rank columns>, yr, month, month_num,
        mean_ret, adj_ret
    """
    merged = compute_raw_features(merged)
    dat = select_modeling_panel(merged)
    dat = apply_cross_sectional_winsorization(dat, winsorize_low, winsorize_high)
    dat = add_percentile_ranks(dat)
    dat = add_date_utilities(dat)
    dat = add_demeaned_target(dat)
    dat = filter_to_coverage_start_year(dat, data_start_year)
    return dat
