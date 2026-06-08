"""
market_model.py
===============
Run CAPM (Capital Asset Pricing Model) regressions for each decile
portfolio and the long-short spread, producing alpha and beta estimates
plus their statistical significance.

The CAPM regression
-------------------
For each portfolio p, we estimate:
    excess_returns_p_t = α_p + β_p * mkt_excess_t + ε_p_t

where:
    excess_returns_p_t = portfolio p's return in month t, minus
                         the risk-free rate (RF) — EXCEPT for the
                         long-short spread, which is already a self-
                         financing position and uses raw RET.
    mkt_excess_t       = market return (Mkt) minus risk-free rate (RF),
                         from the Fama-French monthly factors.
    α_p                = portfolio p's monthly alpha (abnormal return
                         beyond what CAPM predicts).
    β_p                = portfolio p's exposure to the overall market.

The output is a table with one row per portfolio (0..9 and 'diff'),
reporting alpha, beta, both t-statistics, and the regression R².

Why the spread doesn't get RF subtracted
----------------------------------------
The long-short spread is computed as decile_9_return - decile_0_return.
Both deciles are long positions, but in a real long-short portfolio,
the short side's proceeds fund the long side — there's no net cash
outlay and no borrowing at the risk-free rate. The spread is already
in "excess" form. Subtracting RF would double-count the financing cost.

For individual deciles (which are long-only positions), subtracting RF
is standard — that's what makes the regression a CAPM regression rather
than a total-return regression.

Output table column ordering
----------------------------
Portfolios are sorted with 'diff' first, then deciles 0..9 in numeric
order. This matches the notebook's sort_key:
    sorted(ranks, key=lambda x: (x != 'diff', x))
which evaluates 'diff' as (False, 'diff') and others as (True, n),
placing 'diff' first.

Expected fama-french market_riskfree file format
------------------------------------------------
The Market_Riskfree.xlsx file must contain columns:
    year   — calendar year (int)
    month  — calendar month (int 1..12)
    Mkt    — market return (decimal, e.g. 0.0123 for +1.23%)
    RF     — risk-free rate (decimal, monthly)

These match the Fama-French monthly factor data file naming convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# Column ordering convention
SPREAD_LABEL = "diff"

# Output columns of the market model regression table
MARKET_MODEL_OUTPUT_COLS = [
    "rank",
    "alpha_monthly",
    "alpha_tstat",
    "beta",
    "beta_tstat",
    "r2",
]


def load_market_riskfree(path: Path) -> pd.DataFrame:
    """Load the Fama-French market-and-risk-free monthly factors file.

    Parameters
    ----------
    path
        Path to Market_Riskfree.xlsx. Must contain columns:
        year (int), month (int), Mkt (decimal return), RF (decimal rate).

    Returns
    -------
    DataFrame with the same columns, indexed 0..N-1.

    Notes
    -----
    The Fama-French data library publishes these factors monthly. The
    standard download from the library has columns labeled exactly as
    above (with Mkt-RF as a single column, but our local file appears
    to use separate Mkt and RF columns based on the notebook usage).
    """
    logger.info("Loading Fama-French Mkt/RF factors from %s", path)
    mktrf = pd.read_excel(path)
    logger.info("  Loaded %d months of market data", len(mktrf))
    return mktrf


def prepare_regression_inputs(
    meanret_wide: pd.DataFrame,
    mktrf: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the per-(month, portfolio) regression dataset.

    Joins the monthly decile + spread returns with the Fama-French
    market factors on (year, month), computes excess returns for each
    portfolio, and returns a long-format DataFrame ready for the
    per-portfolio OLS regressions.

    Parameters
    ----------
    meanret_wide
        Output of portfolio_construction.run_portfolio_construction()
        — the wide DataFrame with MultiIndex columns ('RET', 0..9, 'diff').
    mktrf
        Output of load_market_riskfree().

    Returns
    -------
    DataFrame with columns:
        year, month, rank (str), RET, Mkt, RF,
        mkt_excess_returns, rank_str, excess_returns

    where excess_returns is:
        - RET - RF for deciles 0..9
        - RET (no subtraction) for the long-short spread
    """
    # Stack back to long format and reset to flat columns.
    # `level=-1` stacks the decile column level (the outer MultiIndex
    # level is 'RET', the inner level is the decile id 0..9 + 'diff').
    portfolio_returns = (
        meanret_wide
        .stack(level=-1, future_stack=True)
        .reset_index()
        [["yr", "month", "rank", "RET"]]
        .copy()
    )

    # Rename yr to year to match the Fama-French file's column name
    portfolio_returns.rename(columns={"yr": "year"}, inplace=True)

    # Inner join: keep only months where we have both portfolio returns
    # and market factor data
    merge_data = pd.merge(
        portfolio_returns, mktrf,
        how="inner",
        on=["year", "month"],
    )

    # Market excess return: Mkt - RF
    merge_data["mkt_excess_returns"] = merge_data["Mkt"] - merge_data["RF"]

    # Cast rank to string so 'diff' and the integer deciles share a column.
    # The integer deciles 0..9 become '0'..'9'.
    merge_data["rank_str"] = merge_data["rank"].astype(str)

    # Portfolio excess returns:
    #   - For the long-short spread ('diff'): use RET directly because
    #     a long-short position is already self-financing (no RF cost).
    #   - For deciles 0..9: subtract RF (standard long-only excess return).
    merge_data["excess_returns"] = np.where(
        merge_data["rank_str"] == SPREAD_LABEL,
        merge_data["RET"],
        merge_data["RET"] - merge_data["RF"],
    )

    logger.info(
        "  Prepared %d (month, portfolio) regression observations across %d months",
        len(merge_data),
        merge_data["month"].nunique() * merge_data["year"].nunique(),
    )

    return merge_data


def run_capm_regression(
    portfolio_data: pd.DataFrame,
    label: str,
) -> dict:
    """Run a CAPM regression for a single portfolio.

    Parameters
    ----------
    portfolio_data
        Long-format DataFrame containing one portfolio's monthly
        excess_returns and mkt_excess_returns columns.
    label
        Identifier for this portfolio ('diff' or '0'..'9').

    Returns
    -------
    Dict with keys: rank, alpha_monthly, alpha_tstat, beta, beta_tstat, r2.

    Notes
    -----
    Uses statsmodels OLS with an explicit constant term (intercept).
    The intercept estimates alpha — the abnormal return beyond what
    CAPM predicts given the portfolio's market beta.
    """
    X = sm.add_constant(portfolio_data["mkt_excess_returns"])
    y = portfolio_data["excess_returns"]
    result = sm.OLS(y, X).fit()

    return {
        "rank": label,
        "alpha_monthly": result.params["const"],
        "alpha_tstat": result.tvalues["const"],
        "beta": result.params["mkt_excess_returns"],
        "beta_tstat": result.tvalues["mkt_excess_returns"],
        "r2": result.rsquared,
    }


def run_capm_regressions_for_all_portfolios(
    merge_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run CAPM regressions for every decile and the spread.

    Iterates over each unique portfolio identifier ('diff' and '0'..'9'),
    runs a CAPM regression, and collects the results into a single table.

    Portfolios are sorted with 'diff' first, then deciles 0..9 in
    numeric order — matching the original notebook's display order.

    Parameters
    ----------
    merge_data
        Output of prepare_regression_inputs().

    Returns
    -------
    DataFrame with columns:
        rank, alpha_monthly, alpha_tstat, beta, beta_tstat, r2
    Sorted with 'diff' first, then deciles 0..9.
    """
    logger.info("Running CAPM regressions for each portfolio")

    # Sort key: 'diff' evaluates as (False, 'diff'), others as (True, x).
    # Boolean False sorts before True, so 'diff' comes first.
    sort_key = lambda x: (x != SPREAD_LABEL, x)
    portfolios = sorted(merge_data["rank_str"].unique(), key=sort_key)

    rows = []
    for portfolio_label in portfolios:
        portfolio_data = merge_data[merge_data["rank_str"] == portfolio_label]
        row = run_capm_regression(portfolio_data, portfolio_label)
        rows.append(row)

        logger.info(
            "  %s: alpha=%.4f (t=%.2f), beta=%.4f (t=%.2f), R²=%.4f",
            portfolio_label,
            row["alpha_monthly"], row["alpha_tstat"],
            row["beta"], row["beta_tstat"],
            row["r2"],
        )

    market_model = pd.DataFrame(rows)[MARKET_MODEL_OUTPUT_COLS]
    return market_model


def run_market_model(
    meanret_wide: pd.DataFrame,
    market_riskfree_path: Path,
) -> pd.DataFrame:
    """Run the full CAPM market model pipeline end-to-end.

    Combines load_market_riskfree(), prepare_regression_inputs(), and
    run_capm_regressions_for_all_portfolios() into a single orchestrated
    call. This is what run_pipeline.py uses.

    Parameters
    ----------
    meanret_wide
        Wide-format decile returns from portfolio_construction.
    market_riskfree_path
        Path to the Fama-French Mkt/RF factor file.

    Returns
    -------
    Market model results table with one row per portfolio.
    """
    mktrf = load_market_riskfree(market_riskfree_path)
    merge_data = prepare_regression_inputs(meanret_wide, mktrf)
    market_model = run_capm_regressions_for_all_portfolios(merge_data)
    return market_model
