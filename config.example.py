"""
config.example.py
=================
Template for local configuration. Copy this file to `config.py` and fill in
your local paths. `config.py` is gitignored — never commit it.

Usage:
    cp config.example.py config.py
    # then edit config.py with your local paths

All pipeline modules import from `config.py`:
    from config import PATHS, BACKTEST_CONFIG, MODEL_PARAMS
"""

from pathlib import Path

# =====================================================================
# PROJECT PATHS
# =====================================================================
# Resolve relative to the repository root regardless of where scripts run.

PROJECT_ROOT = Path(__file__).resolve().parent

PATHS = {
    # Raw CRSP/Compustat data files from WRDS — see data/README.md for
    # instructions on obtaining these. NEVER commit these to a public repo.
    "crsp_returns": PROJECT_ROOT / "data" / "raw" / "CRSP_monthly_returns_1995_2024.csv.gz",
    "compustat_chars": PROJECT_ROOT / "data" / "raw" / "Compustat_characteristics_1995_2025.csv.gz",
    "market_riskfree": PROJECT_ROOT / "data" / "raw" / "Market_Riskfree.xlsx",

    # Pipeline outputs — regenerable, gitignored
    "outputs_dir": PROJECT_ROOT / "outputs",
}


# =====================================================================
# BACKTEST CONFIGURATION
# =====================================================================
# Knobs that control universe construction, training window, and the
# rolling backtest schedule.

BACKTEST_CONFIG = {
    # Data coverage start year (CRSP is loaded from this year forward)
    "data_start_year": 1995,

    # Minimum lagged market cap to include a stock (in thousands of $).
    # Excludes micro-cap shells. Note that CRSP marketcap_raw = SHROUT * PRC
    # is in thousands because SHROUT is in thousands of shares.
    "min_market_cap_thousands": 10_000,

    # Allowed exchange codes (PRIMEXCH): N=NYSE, Q=NASDAQ, A=AMEX
    "allowed_exchanges": ["N", "Q", "A"],

    # Allowed share codes (SHRCD): 10/11/12 are common US equities
    "allowed_share_codes": [10, 11, 12],

    # Months of training data per rolling window
    "rolling_window_months": 60,

    # Year from which to start computing portfolio returns. Portfolios use
    # data from 1997 onward in the report (60-month training + 1997 start).
    # The notebook filters predictions to yr >= 2000 before portfolio
    # construction. We preserve that filter here.
    "portfolio_start_year": 2000,

    # Lag (months) applied to Compustat datadate to ensure no look-ahead.
    # The report uses 5 months; Compustat fiscal year-end data is typically
    # available with a 4-5 month delay due to 10-K filing windows.
    "compustat_lag_months": 5,

    # Maximum time-tolerance (days) for the merge_asof between CRSP and
    # Compustat. 365 = use the most recent Compustat row within 1 year.
    "merge_asof_tolerance_days": 365,
}


# =====================================================================
# XGBOOST HYPERPARAMETERS
# =====================================================================
# These are the production values from the original notebook. They are
# NOT heavily tuned — a more rigorous grid search is on the "future work"
# list documented in docs/methodology.md.

MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 4,
    "min_child_weight": 1,
    "gamma": 0.2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "learning_rate": 0.1,
    "n_estimators": 40,
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 4,
}


# =====================================================================
# WINSORIZATION BOUNDS
# =====================================================================
# Cross-sectional Winsorization is applied per-month to all features and
# the target return. The 1%/99% bounds clip extreme tail values without
# removing observations.

WINSORIZE_BOUNDS = {
    "low": 0.01,
    "high": 0.99,
}
