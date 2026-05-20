# Pipeline Outputs

This directory holds the four artifacts produced by `src/run_pipeline.py`. All four are gitignored because they are fully regenerable from the source data. This README documents the schema of each so a reader of the repo understands what the pipeline produces without needing to run it.

## Summary of files

| Filename | Rows (typical) | Granularity |
|---|---|---|
| `rolling_xgb_pred_returns_project.csv` | ~1,195,000 | One row per (PERMNO, prediction month) |
| `rolling_xgb_r2_project.csv` | ~330 | One row per test month |
| `portfolio_performance_summary.csv` | 11 | Per-decile (0..9) + long-short spread |
| `market_model_results.csv` | 11 | Per-decile (0..9) + long-short spread |

Row counts reflect the validated April 2026 + January 2026 CRSP/Compustat snapshot — slightly different snapshots will produce slightly different totals.

---

## 1. `rolling_xgb_pred_returns_project.csv`

Stock-level XGBoost predictions for every test month in the rolling backtest. This is the rawest output — every other downstream artifact is derived from this one.

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Month-end date of the prediction |
| `PERMNO` | int | CRSP permanent issue identifier |
| `yr` | int | Calendar year |
| `month` | int | Calendar month (1..12) |
| `RET` | float | Realized monthly return for this stock |
| `adj_ret` | float | Cross-sectionally demeaned return (RET − cross-sectional mean for this month) |
| `predicted_adj_ret` | float | XGBoost-predicted demeaned return |

**Typical uses:**
- Compare predicted vs. actual return per stock
- Per-stock attribution: which names contributed most to the spread return?
- Feed into alternative portfolio construction (rank-weighted, vol-weighted, etc.)

**Approximate scale:** ~3,000-4,500 rows per month over 330 test months = ~1.2 million rows total.

---

## 2. `rolling_xgb_r2_project.csv`

Per-month out-of-sample R² of the cross-sectional regression. One row per test month from the first prediction to the last.

| Column | Type | Description |
|---|---|---|
| `date` | datetime | The test month being predicted |
| `r2` | float | Out-of-sample R² for this month |
| `n_test` | int | Stocks in this month's cross-section |

**Typical use:** time-series plot to identify regime breakdowns. Did R² collapse in October 2008? In March 2020? Was the model still working in 2024?

**Expected magnitude:** average R² ≈ 0.008 (0.8%), median ≈ 0.010. This is **normal** for monthly cross-sectional return prediction. The signal is weak per-month but sign-aligned across hundreds of months, which is what drives the Sharpe ratio.

A simpler way to think about it: predicting any given month's cross-sectional return ordering is hard, but you don't need to be right every month — you need to be right *more often than wrong*, consistently, across a long period.

---

## 3. `portfolio_performance_summary.csv`

Aggregate monthly-return statistics for each decile portfolio and the long-short spread.

| Column | Type | Description |
|---|---|---|
| (index) `rank` | str | `'0'`..`'9'` for deciles; `'diff'` for the spread |
| `mean` | float | Average monthly return |
| `std` | float | Standard deviation of monthly returns |
| `t_stat` | float | One-sample t vs zero: `sqrt(n-1) × mean / std` |
| `monthly_sharpe` | float | `mean / std` |
| `annual_sharpe` | float | `sqrt(12) × monthly_sharpe` |

**Validated numbers from this pipeline (300 months: Jan 2000 – Dec 2024):**

| rank | mean | std | t_stat | monthly_sharpe | annual_sharpe |
|---|---:|---:|---:|---:|---:|
| diff | 0.0479 | 0.0655 | 12.64 | 0.731 | **2.53** |
| 9 | 0.0233 | 0.0908 | 4.43 | 0.257 | 0.89 |
| 0 | -0.0247 | 0.0742 | -5.74 | -0.332 | -1.15 |

The long-short spread is the headline portfolio. Its 2.53 annualized Sharpe and 12.64 t-statistic place it far beyond the threshold for statistical significance (typically |t| > 2).

---

## 4. `market_model_results.csv`

CAPM regression results for each portfolio. The regression for portfolio `p` is:

```
excess_returns_p_t = α_p + β_p × mkt_excess_t + ε_p_t
```

For the long-short spread, `excess_returns` is just RET (no risk-free subtraction — it's already a self-financing position). For deciles 0-9 (long-only positions), `excess_returns = RET − RF`.

| Column | Type | Description |
|---|---|---|
| `rank` | str | `'diff'` (sorted first) or `'0'`..`'9'` |
| `alpha_monthly` | float | Monthly alpha from the CAPM regression |
| `alpha_tstat` | float | t-statistic of the alpha estimate |
| `beta` | float | Market beta |
| `beta_tstat` | float | t-statistic of the beta estimate |
| `r2` | float | Regression R² |

**Validated numbers from this pipeline (300 months):**

| rank | alpha_monthly | alpha_tstat | beta | beta_tstat | r2 |
|---|---:|---:|---:|---:|---:|
| diff | **0.0436** | **13.24** | 0.726 | 10.16 | 0.257 |
| 0 | -0.0294 | -10.69 | 1.251 | 20.94 | 0.595 |
| 9 | 0.0142 | 3.13 | 1.977 | 20.11 | 0.576 |

**Why the alpha matters:** the spread's 4.36% monthly alpha (52% annualized) is **what's left after removing market beta exposure**. This is the rigorous answer to "is this strategy just leveraged market exposure?" — no, the CAPM alpha is t=13.24 significant against zero. The signal is genuine cross-sectional alpha, not factor risk.

The decile-level pattern is also informative. Alpha progresses monotonically from -2.94% (decile 0, "loser" stocks) to +1.42% (decile 9, "winner" stocks). This monotonicity is the structural fingerprint of a working predictive model.

---

## How outputs are saved

`src/run_pipeline.py` saves all four files at the end of the run via plain `df.to_csv()`. The portfolio summary keeps its index (`rank` is the index, not a column); the other three use `index=False`. The directory is created if it doesn't exist:

```python
output_dir = PATHS["outputs_dir"]
output_dir.mkdir(parents=True, exist_ok=True)
```

`PATHS["outputs_dir"]` defaults to `<repo_root>/outputs/` per `config.example.py`.

## Regenerating these outputs

Run:
```bash
python src/run_pipeline.py
```

Expected runtime: about 2 minutes on a modern laptop (stage 3 — the 330 rolling XGBoost fits — is the dominant cost).
