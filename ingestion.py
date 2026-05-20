"""
ingestion.py
============
Load and preprocess CRSP and Compustat data, then merge them via an
as-of join with a 5-month lag on Compustat to prevent look-ahead bias.

Pipeline stages:
    1. Load CRSP monthly returns (price, shares outstanding, cumulative
       factor for shares, primary exchange, share code).
    2. CRSP cleanup: compute split-adjusted shares, market cap (lagged
       one period), cohort filter on exchange (NYSE/AMEX/NASDAQ) and
       share code (US common equities only).
    3. Load Compustat annual characteristics (book equity, income before
       extraordinary items, operating cash flow, total assets).
    4. Compustat cleanup: shift the effective date forward by 5 months
       to ensure fiscal-year-end data is publicly available before use.
    5. Merge: as-of backward join with 365-day tolerance, so each CRSP
       month gets matched to the most recent Compustat row available.

The output is a unified panel dataset with one row per PERMNO-month,
ready for feature engineering downstream.

References:
    - Compustat 5-month publication lag: 10-K filings are typically
      due 60-90 days after fiscal year-end, plus a buffer.
    - merge_asof direction='backward': for each CRSP date, find the
      most recent Compustat row with date <= CRSP date.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DateOffset

logger = logging.getLogger(__name__)


# =====================================================================
# CRSP COLUMNS USED
# =====================================================================
# PERMNO        — CRSP permanent issue identifier
# date          — month-end date
# RET           — monthly return (can be string 'B', 'C', etc — coerced)
# PRC           — closing price (negative for bid-ask midpoint; abs() applied)
# SHROUT        — shares outstanding (thousands)
# CFACSHR       — cumulative adjustment factor for splits/dividends
# PRIMEXCH      — primary exchange: N=NYSE, Q=NASDAQ, A=AMEX, others excluded
# SHRCD         — share code: 10/11/12 = US common equities

CRSP_REQUIRED_COLS = [
    "PERMNO", "date", "RET", "PRC", "SHROUT", "CFACSHR", "PRIMEXCH", "SHRCD",
]


# =====================================================================
# COMPUSTAT COLUMNS USED
# =====================================================================
# LPERMNO / PERMNO — link to CRSP via the WRDS linking table (renamed if
#                    only LPERMNO exists in the source file)
# datadate         — fiscal period end date
# ceq              — common/ordinary equity (book equity)
# ib               — income before extraordinary items
# oancf            — operating activities net cash flow
# at               — total assets

COMPUSTAT_REQUIRED_COLS = ["PERMNO", "datadate", "ceq", "ib", "oancf", "at"]


def load_crsp(
    path: Path,
    min_market_cap_thousands: int,
    allowed_exchanges: list[str],
    allowed_share_codes: list[int],
) -> pd.DataFrame:
    """Load CRSP monthly returns and apply universe filters.

    Parameters
    ----------
    path
        Path to the CRSP CSV (gzipped or plain).
    min_market_cap_thousands
        Minimum lagged market cap (thousands of $). Stocks below this
        threshold in the prior month are excluded for the current month.
    allowed_exchanges
        Primary exchange codes to keep (typically N, Q, A).
    allowed_share_codes
        Share codes to keep (typically 10, 11, 12 for US common equities).

    Returns
    -------
    DataFrame with columns:
        PERMNO, PRIMEXCH, date, RET, PRC, SHROUT, CFACSHR,
        marketcap, SHROUT_adj

    Notes
    -----
    The market cap filter is applied to the *lagged* market cap (prior
    month) rather than the contemporaneous value. This ensures the
    inclusion decision uses information available at the start of the
    prediction month.
    """
    logger.info("Loading CRSP returns from %s", path)
    returns = pd.read_csv(path)
    logger.info("  Raw rows: %s", f"{len(returns):,}")

    # Split-adjusted shares outstanding (used for new_issue feature later)
    returns["SHROUT_adj"] = returns["SHROUT"] / returns["CFACSHR"]

    # CRSP records bid-ask midpoint as negative; take absolute value
    returns["PRC"] = returns["PRC"].abs()

    # RET sometimes encoded as string flags ('B', 'C', etc.) — coerce non-numeric to NaN
    returns["RET"] = pd.to_numeric(returns["RET"], errors="coerce")

    # Raw market cap (in thousands $, since SHROUT is in thousands)
    returns["marketcap_raw"] = returns["SHROUT"] * returns["PRC"]

    # Lag market cap by one period per PERMNO — inclusion criterion uses
    # prior month's market cap to avoid look-ahead bias
    returns["marketcap"] = returns.groupby("PERMNO")["marketcap_raw"].shift()

    # Apply minimum market cap filter (excludes micro-cap shells)
    returns["marketcap"] = np.where(
        returns["marketcap"] < min_market_cap_thousands,
        np.nan,
        returns["marketcap"],
    )

    # Cohort filter: US common equities on NYSE/AMEX/NASDAQ
    returns = returns[
        returns["PRIMEXCH"].isin(allowed_exchanges)
        & returns["SHRCD"].isin(allowed_share_codes)
    ].copy()

    # Type/format cleanup
    returns["date"] = pd.to_datetime(returns["date"])

    # Drop rows missing any required value (RET, marketcap, etc.)
    keep_cols = [
        "PERMNO", "PRIMEXCH", "date", "RET", "PRC", "SHROUT",
        "CFACSHR", "marketcap", "SHROUT_adj",
    ]
    returns = returns[keep_cols].dropna().copy()
    returns["PERMNO"] = returns["PERMNO"].astype(int)
    returns = returns.sort_values(["PERMNO", "date"]).reset_index(drop=True)

    logger.info("  After universe + dropna filter: %s rows", f"{len(returns):,}")
    return returns


def load_compustat(
    path: Path,
    lag_months: int,
) -> pd.DataFrame:
    """Load Compustat annual characteristics with publication lag.

    Parameters
    ----------
    path
        Path to the Compustat CSV (gzipped or plain).
    lag_months
        Number of months to shift datadate forward by, simulating the
        delay between fiscal year-end and public availability via the
        10-K filing. Typical: 5 months.

    Returns
    -------
    DataFrame with columns:
        PERMNO, datadate, ceq, ib, oancf, at, date, lag_at
    where:
        date = datadate + lag_months
        lag_at = previous year's `at` per PERMNO

    Notes
    -----
    Compustat sometimes exposes the link key as LPERMNO (from the WRDS
    CRSP-Compustat linking table) rather than PERMNO directly. We handle
    both naming conventions.
    """
    logger.info("Loading Compustat characteristics from %s", path)
    cstat = pd.read_csv(path)
    logger.info("  Raw rows: %s", f"{len(cstat):,}")

    # Some WRDS exports use LPERMNO (linked PERMNO from the CRSP-Compustat
    # link table) — rename to PERMNO for consistency.
    if "LPERMNO" in cstat.columns and "PERMNO" not in cstat.columns:
        cstat.rename(columns={"LPERMNO": "PERMNO"}, inplace=True)

    cstat = cstat[COMPUSTAT_REQUIRED_COLS].dropna().copy()
    cstat["PERMNO"] = cstat["PERMNO"].astype(int)
    cstat["datadate"] = pd.to_datetime(cstat["datadate"])

    # Shift the effective availability date forward by lag_months.
    # This is the single most important leakage-prevention step:
    # Compustat fiscal year-end data is not publicly available until
    # the 10-K is filed, typically 4-5 months later.
    cstat["date"] = cstat["datadate"] + DateOffset(months=lag_months)
    cstat = cstat.sort_values(["PERMNO", "date"]).reset_index(drop=True)

    # Lagged total assets, used for the `investment` feature downstream
    cstat["lag_at"] = cstat.groupby("PERMNO")["at"].shift()

    logger.info("  After dropna filter: %s rows", f"{len(cstat):,}")
    return cstat


def merge_crsp_compustat(
    returns: pd.DataFrame,
    cstat: pd.DataFrame,
    tolerance_days: int,
) -> pd.DataFrame:
    """As-of merge CRSP monthly returns with the most recent Compustat row.

    For each (PERMNO, date) in CRSP, finds the most recent Compustat row
    for the same PERMNO with effective date <= CRSP date, within the
    specified tolerance window.

    Parameters
    ----------
    returns
        Cleaned CRSP DataFrame from load_crsp().
    cstat
        Cleaned Compustat DataFrame from load_compustat() — date column
        must already include the publication lag.
    tolerance_days
        Maximum days between CRSP date and Compustat effective date.
        Typical: 365 (use Compustat data up to one year old).

    Returns
    -------
    Merged DataFrame, one row per (PERMNO, CRSP date) where a matching
    Compustat row was found within tolerance, with rows dropped for any
    nulls remaining.
    """
    logger.info(
        "Merging CRSP and Compustat with %d-day tolerance, backward direction",
        tolerance_days,
    )

    merged = pd.merge_asof(
        returns.sort_values("date"),
        cstat.sort_values("date"),
        by="PERMNO",
        left_on="date",
        right_on="date",
        tolerance=dt.timedelta(days=tolerance_days),
        direction="backward",
    ).dropna().sort_values(["PERMNO", "date"]).reset_index(drop=True)

    logger.info("  Merged rows: %s", f"{len(merged):,}")
    return merged


def run_ingestion(
    crsp_path: Path,
    compustat_path: Path,
    min_market_cap_thousands: int,
    allowed_exchanges: list[str],
    allowed_share_codes: list[int],
    compustat_lag_months: int,
    merge_asof_tolerance_days: int,
) -> pd.DataFrame:
    """Run the full ingestion pipeline end-to-end.

    Combines load_crsp(), load_compustat(), and merge_crsp_compustat()
    into a single orchestrated call. This is what run_pipeline.py uses.

    Returns
    -------
    The fully-merged DataFrame ready for feature engineering.
    """
    returns = load_crsp(
        path=crsp_path,
        min_market_cap_thousands=min_market_cap_thousands,
        allowed_exchanges=allowed_exchanges,
        allowed_share_codes=allowed_share_codes,
    )
    cstat = load_compustat(
        path=compustat_path,
        lag_months=compustat_lag_months,
    )
    merged = merge_crsp_compustat(
        returns=returns,
        cstat=cstat,
        tolerance_days=merge_asof_tolerance_days,
    )
    return merged