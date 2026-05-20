"""
portfolio_construction.py
=========================
Convert XGBoost-predicted cross-sectional return scores into a long-short
decile portfolio backtest, then compute per-decile and long-short
performance statistics.

Methodology
-----------
For each (year, month) cohort of predictions:
    1. Rank stocks by predicted adjusted return (1 = lowest, N = highest)
    2. Sort the rank into 10 equal-population buckets (deciles 0..9)
       where decile 0 = bottom 10%, decile 9 = top 10%
    3. Average the RAW (not demeaned) monthly returns within each decile
       to get 10 portfolio-level return time series
    4. Form the long-short spread:
            spread_t = decile_9_t - decile_0_t
       This is the implementable trading strategy: long the top
       predicted-return decile, short the bottom.

Why decile 9 minus decile 0?
----------------------------
A successful predictive model should rank stocks such that the highest
predicted-return decile (9) outperforms the lowest (0) on average. The
9-minus-0 spread is the standard "long-short" factor portfolio that
isolates the cross-sectional signal from market beta.

Why filter to yr >= 2000?
-------------------------
The rolling-window backtest starts producing predictions ~5 years after
the data start. With CRSP coverage from 1995 and a 60-month training
window, the first valid prediction is for February 2000. We filter to
yr >= 2000 to align the portfolio backtest with the report's "27-year
out-of-sample" claim covering July 2000 through December 2024.

The notebook uses a hardcoded `yr >= 2000` filter; we parametrize it
via portfolio_start_year so it's auditable.

Statistical tests
-----------------
For each decile (and the spread), we compute:
    - mean monthly return
    - std of monthly returns
    - one-sample t-statistic vs zero:  sqrt(n - 1) * mean / std
    - monthly Sharpe ratio:             mean / std
    - annualized Sharpe ratio:          sqrt(12) * monthly_sharpe

The t-statistic uses (n-1) degrees of freedom following the standard
one-sample t-test formula. The annualization assumes monthly returns
are serially uncorrelated, which is the standard assumption for
monthly factor portfolios.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Decile naming — kept as integers 0..9 to match notebook output and
# pd.qcut behavior. Decile 0 is the lowest predicted return, decile 9
# is the highest.
N_DECILES = 10
LONG_DECILE = 9       # top predicted return
SHORT_DECILE = 0      # bottom predicted return
SPREAD_LABEL = "diff" # column name for long-short spread


def assign_deciles(
    predicted_ret_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each prediction to a within-month decile bucket.

    For each (year, month) cohort:
        1. Rank stocks by predicted_adj_ret (1 = lowest)
        2. Sort the rank into 10 equal-population buckets (0..9)

    The two-step rank-then-qcut approach (rather than qcut directly on
    predicted_adj_ret) is the notebook's approach. It produces identical
    bucket assignments in the absence of ties, and uses `rank(method='first')`
    to break ties deterministically by row order when ties exist.

    Parameters
    ----------
    predicted_ret_df
        Output of modeling.run_modeling()['predicted_ret_df']. Must
        contain columns: yr, month, predicted_adj_ret.

    Returns
    -------
    DataFrame with two new columns added:
        rank_order — 1..N rank within (yr, month), ties broken by row order
        rank       — 0..9 decile bucket within (yr, month)
    """
    logger.info("Assigning deciles within each (yr, month) cohort")

    predicted_ret_df = predicted_ret_df.copy()

    # Step 1: Rank predicted returns within each month. Using
    # method='first' (not 'average' or 'min') makes the rank a
    # 1, 2, 3, ... integer sequence with ties broken by row position.
    predicted_ret_df["rank_order"] = (
        predicted_ret_df
        .groupby(["yr", "month"])["predicted_adj_ret"]
        .rank(method="first")
    )

    # Step 2: Sort the rank into 10 equal-population buckets.
    # pd.qcut with labels=False returns integer codes 0..9.
    predicted_ret_df["rank"] = (
        predicted_ret_df
        .groupby(["yr", "month"])["rank_order"]
        .transform(lambda x: pd.qcut(x, N_DECILES, labels=False))
    )

    return predicted_ret_df


def filter_to_portfolio_start_year(
    predicted_ret_df: pd.DataFrame,
    portfolio_start_year: int,
) -> pd.DataFrame:
    """Restrict the backtest to predictions from the given start year.

    Parameters
    ----------
    predicted_ret_df
        Output of assign_deciles().
    portfolio_start_year
        Minimum calendar year (inclusive) to include in the backtest.

    Returns
    -------
    Filtered DataFrame.
    """
    n_before = len(predicted_ret_df)
    out = predicted_ret_df[predicted_ret_df["yr"] >= portfolio_start_year].copy()
    logger.info(
        "  Filtered to yr >= %d: %s -> %s rows",
        portfolio_start_year, f"{n_before:,}", f"{len(out):,}",
    )
    return out


def compute_decile_portfolio_returns(
    pred_filtered: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly equal-weighted returns for each decile + the spread.

    For each (yr, month, decile), averages the raw RET column to get
    the equal-weighted decile portfolio return for that month. Then
    pivots wide and computes the long-short spread (decile 9 - decile 0).

    Parameters
    ----------
    pred_filtered
        Output of filter_to_portfolio_start_year(). Must have columns
        yr, month, rank (the decile), RET.

    Returns
    -------
    DataFrame in WIDE format with one row per (yr, month) and columns:
        ('RET', 0), ('RET', 1), ..., ('RET', 9), ('RET', 'diff')
    Where 'diff' is the long-short spread (decile 9 - decile 0).

    Notes
    -----
    The MultiIndex column structure is a side effect of the pivot. It is
    preserved here to match the notebook's downstream stacking pattern
    used by compute_decile_statistics().
    """
    logger.info("Computing decile portfolio returns")

    # Long-format: one row per (yr, month, decile)
    meanret = (
        pred_filtered
        .groupby(["yr", "month", "rank"])["RET"]
        .mean()
        .to_frame()
    )

    # Pivot to wide: deciles become columns under the ('RET', d) MultiIndex
    meanret = meanret.unstack(level=-1).copy()

    # Long-short spread: top decile (9) minus bottom decile (0)
    meanret[("RET", SPREAD_LABEL)] = (
        meanret[("RET", LONG_DECILE)] - meanret[("RET", SHORT_DECILE)]
    )

    logger.info(
        "  Portfolio period: %d months, %d decile columns + spread",
        len(meanret), N_DECILES,
    )

    return meanret


def compute_decile_statistics(
    meanret_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-decile performance statistics.

    Parameters
    ----------
    meanret_wide
        Output of compute_decile_portfolio_returns() — wide format with
        MultiIndex columns ('RET', 0..9, 'diff').

    Returns
    -------
    DataFrame indexed by decile (0..9 plus 'diff') with columns:
        mean            — average monthly return
        std             — std dev of monthly returns
        t_stat          — one-sample t vs zero: sqrt(n-1) * mean / std
        monthly_sharpe  — mean / std
        annual_sharpe   — sqrt(12) * monthly_sharpe

    Notes
    -----
    The t-statistic uses (n - 1) in the radical following the standard
    one-sample t-test formula. The annualization assumes monthly returns
    are serially uncorrelated.
    """
    n_months = len(meanret_wide)

    # Stack back to long format so each row is one (month, decile)
    # observation, then group by decile to compute statistics.
    # future_stack=True suppresses a pandas 2.x deprecation warning
    # for the existing stack behavior we want.
    meanret_stack = meanret_wide.stack(level=-1, future_stack=True).copy()

    stats = (
        meanret_stack
        .groupby("rank")["RET"]
        .agg(["mean", "std"])
    )

    # One-sample t-stat against zero
    stats["t_stat"] = np.sqrt(n_months - 1) * stats["mean"] / stats["std"]

    # Sharpe ratios (monthly and annualized)
    stats["monthly_sharpe"] = stats["mean"] / stats["std"]
    stats["annual_sharpe"] = np.sqrt(12) * stats["monthly_sharpe"]

    logger.info(
        "  Computed stats for %d portfolios (n=%d months)",
        len(stats), n_months,
    )

    return stats


def get_long_short_spread_series(
    meanret_wide: pd.DataFrame,
) -> pd.Series:
    """Extract the long-short spread as a flat-indexed time series.

    Parameters
    ----------
    meanret_wide
        Output of compute_decile_portfolio_returns().

    Returns
    -------
    Series indexed by (yr, month) with the spread return for each month.
    This is the input to market_model.py for the CAPM regression.
    """
    spread = meanret_wide[("RET", SPREAD_LABEL)]
    return spread


def run_portfolio_construction(
    predicted_ret_df: pd.DataFrame,
    portfolio_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run the full portfolio construction pipeline end-to-end.

    Combines decile assignment, year filtering, portfolio return
    computation, and statistics into a single orchestrated call.

    Parameters
    ----------
    predicted_ret_df
        Output of modeling.run_modeling()[0] — long-format predictions.
    portfolio_start_year
        Minimum year (inclusive) for the portfolio backtest.

    Returns
    -------
    decile_stats : DataFrame
        Per-decile statistics table (10 deciles + spread).
    meanret_wide : DataFrame
        Wide-format monthly decile returns.
    spread_series : Series
        Long-short monthly return series, ready for CAPM regression.
    """
    predicted_ret_df = assign_deciles(predicted_ret_df)
    pred_filtered = filter_to_portfolio_start_year(
        predicted_ret_df, portfolio_start_year
    )
    meanret_wide = compute_decile_portfolio_returns(pred_filtered)
    decile_stats = compute_decile_statistics(meanret_wide)
    spread_series = get_long_short_spread_series(meanret_wide)
    return decile_stats, meanret_wide, spread_series