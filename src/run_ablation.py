"""
run_ablation.py
===============
Feature-ablation harness for the rolling XGBoost factor model.

Goal
----
Isolate how much of the headline performance comes from the two
SELF-PROPOSED signals (reversal_1m, vol_12m) versus the standard
academic / control features.

It does this by holding EVERYTHING constant -- same universe, same dates,
same 60-month rolling window, same Winsorization, same percentile-rank
transform, same XGBoost params, same decile construction, same CAPM
regression -- and changing ONLY the feature list passed to the model:

    baseline_no_proposed : all features EXCEPT reversal_1m, vol_12m
    full_model           : all 10 features (reproduces the published run)
    proposed_only        : reversal_1m, vol_12m only (sanity check)

The gap between baseline_no_proposed and full_model is the number that
belongs on the resume: the lift your own research added.

Ingestion + feature engineering run ONCE (they are identical across
feature sets); only modeling onward is repeated per feature set.

To run (from repo root, same as run_pipeline):
    python src/run_ablation.py

Requires the same config.py that run_pipeline.py uses.

IMPORTANT before trusting the deltas
-------------------------------------
1. Make sure MODEL_PARAMS in config.py sets a fixed seed
   (e.g. random_state=42). XGBoost with subsampling is stochastic; an
   unfixed seed will add noise to the delta and contaminate attribution.
2. Confirm `full_model` below reproduces your published headline
   (annualized Sharpe ~2.52, monthly alpha ~+4.43%). If it does not,
   this harness differs from your original run and the baseline is not
   yet a valid comparison -- stop and reconcile before reporting.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Make repo root + src importable regardless of invocation dir,
# mirroring run_pipeline.py.
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
from feature_engineering import run_feature_engineering, FEATURES
from modeling import run_rolling_backtest
from portfolio_construction import run_portfolio_construction
from market_model import run_market_model

logger = logging.getLogger(__name__)


# =====================================================================
# FEATURE SETS
# =====================================================================
# Your two self-proposed signals (per the report / resume framing):
#   reversal_1m -> short-term reversal
#   vol_12m     -> idiosyncratic-volatility signal
PROPOSED = ["reversal_1m", "vol_12m"]

BASELINE = [f for f in FEATURES if f not in PROPOSED]   # the other 8

FEATURE_SETS: dict[str, list[str]] = {
    "baseline_no_proposed": BASELINE,        # 8 features
    "full_model":           list(FEATURES),  # 10 features (published run)
    "proposed_only":        PROPOSED,        # 2 features (sanity check)
}


def _pct_rank_cols(raw_features: list[str]) -> list[str]:
    """Map raw feature names to the _pct_rank columns the model consumes."""
    return [f"{f}_pct_rank" for f in raw_features]


def run_one_feature_set(name: str, raw_features: list[str], dat: pd.DataFrame) -> dict:
    """Run modeling -> portfolio -> CAPM for a single feature set.

    Reuses the already-feature-engineered panel `dat`; only the feature
    columns handed to the model change.
    """
    logger.info("=" * 70)
    logger.info("FEATURE SET: %s  (%d features)", name, len(raw_features))
    logger.info("  %s", ", ".join(raw_features))
    logger.info("=" * 70)

    feature_cols = _pct_rank_cols(raw_features)

    predicted_ret_df, _r2_df = run_rolling_backtest(
        dat=dat,
        feature_cols=feature_cols,
        target_col="adj_ret",
        rolling_window_months=BACKTEST_CONFIG["rolling_window_months"],
        model_params=MODEL_PARAMS,
    )

    decile_stats, meanret_wide, _spread_series = run_portfolio_construction(
        predicted_ret_df=predicted_ret_df,
        portfolio_start_year=BACKTEST_CONFIG["portfolio_start_year"],
    )

    market_model = run_market_model(
        meanret_wide=meanret_wide,
        market_riskfree_path=PATHS["market_riskfree"],
    )

    spread = decile_stats.loc["diff"]
    capm = market_model[market_model["rank"] == "diff"].iloc[0]

    return {
        "feature_set":    name,
        "n_features":     len(feature_cols),
        "n_months":       int(len(meanret_wide)),
        "monthly_mean":   float(spread["mean"]),
        "monthly_sharpe": float(spread["monthly_sharpe"]),
        "annual_sharpe":  float(spread["annual_sharpe"]),
        "spread_tstat":   float(spread["t_stat"]),
        "alpha_monthly":  float(capm["alpha_monthly"]),
        "alpha_tstat":    float(capm["alpha_tstat"]),
        "beta":           float(capm["beta"]),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.time()

    # ---- Shared stages: run ONCE -------------------------------------
    merged = run_ingestion(
        crsp_path=PATHS["crsp_returns"],
        compustat_path=PATHS["compustat_chars"],
        min_market_cap_thousands=BACKTEST_CONFIG["min_market_cap_thousands"],
        allowed_exchanges=BACKTEST_CONFIG["allowed_exchanges"],
        allowed_share_codes=BACKTEST_CONFIG["allowed_share_codes"],
        compustat_lag_months=BACKTEST_CONFIG["compustat_lag_months"],
        merge_asof_tolerance_days=BACKTEST_CONFIG["merge_asof_tolerance_days"],
    )
    dat = run_feature_engineering(
        merged=merged,
        winsorize_low=WINSORIZE_BOUNDS["low"],
        winsorize_high=WINSORIZE_BOUNDS["high"],
        data_start_year=BACKTEST_CONFIG["data_start_year"],
    )

    # ---- Per-feature-set stages --------------------------------------
    rows = [run_one_feature_set(name, feats, dat)
            for name, feats in FEATURE_SETS.items()]
    results = pd.DataFrame(rows).set_index("feature_set")

    # ---- Deltas (the resume number) ----------------------------------
    base = results.loc["baseline_no_proposed"]
    full = results.loc["full_model"]

    sharpe_lift_abs = full["annual_sharpe"] - base["annual_sharpe"]
    sharpe_lift_pct = (full["annual_sharpe"] / base["annual_sharpe"] - 1) * 100
    alpha_lift_abs = full["alpha_monthly"] - base["alpha_monthly"]

    out_dir = PATHS["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "ablation_results.csv")

    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY  (spread = decile 9 - decile 0)")
    logger.info("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        logger.info("\n%s", results[[
            "n_features", "annual_sharpe", "spread_tstat",
            "alpha_monthly", "alpha_tstat", "beta",
        ]].to_string())

    logger.info("-" * 70)
    logger.info("LIFT FROM PROPOSED SIGNALS (reversal_1m + vol_12m):")
    logger.info("  Annualized Sharpe: %.2f  ->  %.2f   (%+.2f, %+.1f%%)",
                base["annual_sharpe"], full["annual_sharpe"],
                sharpe_lift_abs, sharpe_lift_pct)
    logger.info("  Monthly alpha:     %+.4f -> %+.4f  (%+.4f)",
                base["alpha_monthly"], full["alpha_monthly"], alpha_lift_abs)
    logger.info("  (full alpha t-stat=%.2f, baseline alpha t-stat=%.2f)",
                full["alpha_tstat"], base["alpha_tstat"])
    logger.info("=" * 70)
    logger.info("Saved: %s", out_dir / "ablation_results.csv")
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()