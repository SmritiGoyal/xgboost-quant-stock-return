"""
run_pipeline.py
===============
End-to-end orchestrator for the XGBoost quant pipeline.

Runs all five stages in order:
    1. Ingestion          — CRSP + Compustat load + as-of merge
    2. Feature engineering — 10 cross-sectional features
    3. Modeling            — rolling-window XGBoost training
    4. Portfolio construction — decile sort + long-short spread
    5. Market model        — CAPM regressions per portfolio

Saves four output artifacts to the configured outputs directory:
    - rolling_xgb_pred_returns_project.csv  (one row per stock-month
                                             prediction)
    - rolling_xgb_r2_project.csv            (per-month out-of-sample R²)
    - portfolio_performance_summary.csv     (per-decile + spread stats)
    - market_model_results.csv              (CAPM alpha/beta table)

To run:
    python -m src.run_pipeline

Or directly:
    python src/run_pipeline.py

Configuration is read from config.py at the repository root. Copy
config.example.py to config.py first and fill in your local paths.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Make the repo root importable so `config.py` resolves regardless
# of where this script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from config import (
        PATHS,
        BACKTEST_CONFIG,
        MODEL_PARAMS,
        WINSORIZE_BOUNDS,
    )
except ImportError as exc:
    raise ImportError(
        "Missing config.py. Copy config.example.py to config.py at the "
        "repo root and fill in your local paths."
    ) from exc

from ingestion import run_ingestion
from feature_engineering import run_feature_engineering
from modeling import run_modeling
from portfolio_construction import run_portfolio_construction
from market_model import run_market_model


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure root logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _format_seconds(seconds: float) -> str:
    """Human-friendly elapsed-time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _stage_header(stage_num: int, total: int, name: str) -> None:
    """Print a visible stage delimiter."""
    bar = "=" * 70
    logger.info(bar)
    logger.info("STAGE %d/%d  %s", stage_num, total, name.upper())
    logger.info(bar)


def run_full_pipeline() -> None:
    """Execute the full pipeline end-to-end."""
    pipeline_start = time.time()
    output_dir = PATHS["outputs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    total_stages = 5

    # =================================================================
    # STAGE 1 — INGESTION
    # =================================================================
    _stage_header(1, total_stages, "Ingestion (CRSP + Compustat + as-of merge)")
    t0 = time.time()

    merged = run_ingestion(
        crsp_path=PATHS["crsp_returns"],
        compustat_path=PATHS["compustat_chars"],
        min_market_cap_thousands=BACKTEST_CONFIG["min_market_cap_thousands"],
        allowed_exchanges=BACKTEST_CONFIG["allowed_exchanges"],
        allowed_share_codes=BACKTEST_CONFIG["allowed_share_codes"],
        compustat_lag_months=BACKTEST_CONFIG["compustat_lag_months"],
        merge_asof_tolerance_days=BACKTEST_CONFIG["merge_asof_tolerance_days"],
    )

    logger.info("Stage 1 done in %s", _format_seconds(time.time() - t0))

    # =================================================================
    # STAGE 2 — FEATURE ENGINEERING
    # =================================================================
    _stage_header(2, total_stages, "Feature engineering (10 features)")
    t0 = time.time()

    dat = run_feature_engineering(
        merged=merged,
        winsorize_low=WINSORIZE_BOUNDS["low"],
        winsorize_high=WINSORIZE_BOUNDS["high"],
        data_start_year=BACKTEST_CONFIG["data_start_year"],
    )

    logger.info("Stage 2 done in %s", _format_seconds(time.time() - t0))

    # =================================================================
    # STAGE 3 — MODELING (rolling-window XGBoost)
    # =================================================================
    _stage_header(3, total_stages, "Modeling (rolling-window XGBoost)")
    t0 = time.time()

    predicted_ret_df, r2_df = run_modeling(
        dat=dat,
        rolling_window_months=BACKTEST_CONFIG["rolling_window_months"],
        model_params=MODEL_PARAMS,
    )

    logger.info("Stage 3 done in %s", _format_seconds(time.time() - t0))

    # =================================================================
    # STAGE 4 — PORTFOLIO CONSTRUCTION
    # =================================================================
    _stage_header(4, total_stages, "Portfolio construction (decile sort + spread)")
    t0 = time.time()

    decile_stats, meanret_wide, spread_series = run_portfolio_construction(
        predicted_ret_df=predicted_ret_df,
        portfolio_start_year=BACKTEST_CONFIG["portfolio_start_year"],
    )

    logger.info("Stage 4 done in %s", _format_seconds(time.time() - t0))

    # =================================================================
    # STAGE 5 — MARKET MODEL (CAPM regressions)
    # =================================================================
    _stage_header(5, total_stages, "Market model (CAPM per portfolio)")
    t0 = time.time()

    market_model = run_market_model(
        meanret_wide=meanret_wide,
        market_riskfree_path=PATHS["market_riskfree"],
    )

    logger.info("Stage 5 done in %s", _format_seconds(time.time() - t0))

    # =================================================================
    # SAVE OUTPUTS
    # =================================================================
    _stage_header(5, total_stages, "Saving outputs")
    logger.info("Output directory: %s", output_dir)

    out_files = {
        "rolling_xgb_pred_returns_project.csv": (predicted_ret_df, False),
        "rolling_xgb_r2_project.csv": (r2_df, False),
        "portfolio_performance_summary.csv": (decile_stats, True),  # keep index
        "market_model_results.csv": (market_model, False),
    }

    for filename, (df, keep_index) in out_files.items():
        path = output_dir / filename
        df.to_csv(path, index=keep_index)
        logger.info("  saved %s  (%d rows)", filename, len(df))

    # =================================================================
    # HEADLINE METRICS — printed for quick eyeball validation
    # =================================================================
    logger.info("=" * 70)
    logger.info("HEADLINE METRICS")
    logger.info("=" * 70)

    spread_stats = decile_stats.loc[decile_stats.index == "diff"]
    if not spread_stats.empty:
        s = spread_stats.iloc[0]
        logger.info("Long-short spread (decile 9 - decile 0):")
        logger.info("  Monthly mean return:    %+.4f", s["mean"])
        logger.info("  Monthly std:            %.4f", s["std"])
        logger.info("  Monthly Sharpe:         %.4f", s["monthly_sharpe"])
        logger.info("  Annualized Sharpe:      %.4f", s["annual_sharpe"])
        logger.info("  t-statistic vs zero:    %.2f", s["t_stat"])

    diff_row = market_model[market_model["rank"] == "diff"]
    if not diff_row.empty:
        m = diff_row.iloc[0]
        logger.info("CAPM regression for spread:")
        logger.info("  Monthly alpha:          %+.4f", m["alpha_monthly"])
        logger.info("  Alpha t-stat:           %.2f", m["alpha_tstat"])
        logger.info("  Market beta:            %.4f", m["beta"])
        logger.info("  Beta t-stat:            %.2f", m["beta_tstat"])
        logger.info("  R²:                     %.4f", m["r2"])

    logger.info("=" * 70)
    logger.info("Total pipeline time: %s", _format_seconds(time.time() - pipeline_start))


if __name__ == "__main__":
    _configure_logging()
    run_full_pipeline()