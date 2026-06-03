# Pipeline Outputs

This directory holds the four artifacts produced by `src/run_pipeline.py`. All four are gitignored because they are fully regenerable from the source data. This README documents the schema of each so a reader of the repo understands what the pipeline produces without needing to run it.

> **Numbers below are the corrected, leak-free outputs.** An earlier run reported an annualized Sharpe of ~2.5; a look-ahead bias in the `vol_12m` feature was found and fixed (see `docs/leakage_audit.md`). The validated tables in this document reflect the corrected pipeline.

## Summary of files

| Filename | Rows (typical) | Granularity |
|---|---|---|
| `rolling_xgb_pred_returns_project.csv` | ~1,195,000 | One row per (PERMNO, prediction month) |
| `rolling_xgb_r2_project.csv` | 330 | One row per test month |
| `portfolio_performance_summary.csv` | 11 | Per-decile (0..9) + long-short spread |
| `market_model_results.csv` | 11 | Per-decile (0..9) + long-short spread |

Row counts reflect the validated CRSP/Compustat snapshot; slightly different snapshots will produce slightly different totals.

---

## 1. `rolling_xgb_pred_returns_project.csv`

Stock-level XGBoost predictions for every test month in the rolling backtest. This is the rawest output — every downstream artifact is derived from it.

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

**Approximate scale:** ~3,000–4,500 rows per month over 330 test months ≈ 1.2 million rows total.

---

## 2. `rolling_xgb_r2_project.csv`

Per-month out-of-sample R² of the cross-sectional regression. One row per test month.

| Column | Type | Description |
|---|---|---|
| `date` | datetime | The test month being predicted |
| `r2` | float | Out-of-sample R² for this month |
| `n_test` | int | Stocks in this month's cross-section |

**Typical use:** time-series plot to identify regime breakdowns (Oct 2008? Mar 2020? still working in 2024?).

**Expected magnitude (corrected):** average R² ≈ 0.0028, median ≈ 0.0025, positive in ≈56% of months. This is **normal** for monthly cross-sectional return prediction — the signal is weak per-month but sign-aligned often enough across hundreds of months to drive the Sharpe. (The earlier leaked run showed a higher average R² ≈ 0.008; that was inflated by the look-ahead bias, since corrected.)

A simpler framing: predicting any single month's cross-sectional ordering is hard, but you don't need to be right every month — you need to be right more often than wrong, consistently, across a long period.

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

**Validated numbers from the corrected pipeline (300 months, 2000–2024):**

| rank | mean | std | t_stat | monthly_sharpe | annual_sharpe |
|---|---:|---:|---:|---:|---:|
| diff | 0.0193 | 0.0648 | 5.15 | 0.298 | **1.03** |
| 9 | 0.0125 | 0.0708 | 3.06 | 0.177 | 0.61 |
| 0 | −0.0068 | 0.0983 | −1.19 | −0.069 | −0.24 |

The long-short spread is the headline portfolio. Its 1.03 annualized Sharpe and 5.15 t-statistic place it well beyond the threshold for statistical significance (typically |t| > 2). Note that Decile 0's raw mean return is not significant on its own (t = −1.19); its contribution to the spread is an alpha (beta-adjusted) effect — see the market model below.

---

## 4. `market_model_results.csv`

CAPM regression results for each portfolio:

```
excess_returns_p_t = α_p + β_p × mkt_excess_t + ε_p_t
```

For the long-short spread, `excess_returns` is RET (self-financing, no risk-free subtraction). For deciles 0–9 (long-only), `excess_returns = RET − RF`.

| Column | Type | Description |
|---|---|---|
| `rank` | str | `'diff'` (sorted first) or `'0'`..`'9'` |
| `alpha_monthly` | float | Monthly alpha from the CAPM regression |
| `alpha_tstat` | float | t-statistic of the alpha estimate |
| `beta` | float | Market beta |
| `beta_tstat` | float | t-statistic of the beta estimate |
| `r2` | float | Regression R² |

**Validated numbers from the corrected pipeline (300 months):**

| rank | alpha_monthly | alpha_tstat | beta | beta_tstat | r2 |
|---|---:|---:|---:|---:|---:|
| diff | **0.0219** | **6.08** | −0.433 | −5.55 | 0.094 |
| 0 | −0.0178 | −4.68 | 1.609 | 19.48 | 0.560 |
| 9 | 0.0041 | 1.51 | 1.176 | 20.12 | 0.576 |

**Why the alpha matters:** the spread's +2.19% monthly alpha (~26% annualized) is what remains after removing market-beta exposure. This is the rigorous answer to "is this just leveraged market exposure?" — no: the CAPM alpha is significant (t = 6.08) against zero, and the spread's beta is actually **negative** (−0.43), so the strategy is net-short the market rather than leveraged-long. The signal is genuine cross-sectional alpha.

The decile-level pattern is broadly monotonic from −1.78% (Decile 0, "losers") up through the middle deciles, flattening at the top (Decile 9 = +0.41%). The short leg is strongly significant (t = −4.68); the long leg is positive but not significant (t = +1.51). The spread's significance is driven mainly by the short side. (The earlier leaked run showed Decile 0 = −2.94% / Decile 9 = +1.42% with a positive spread beta of 0.73 — all inflated by the leak.)

---

## How outputs are saved

`src/run_pipeline.py` saves all four files at the end of the run via plain `df.to_csv()`. The portfolio summary keeps its index (`rank` is the index); the other three use `index=False`. The directory is created if it doesn't exist:

```python
output_dir = PATHS["outputs_dir"]
output_dir.mkdir(parents=True, exist_ok=True)
```

`PATHS["outputs_dir"]` defaults to `<repo_root>/outputs/` per `config.example.py`.

## Regenerating these outputs

```bash
python src/run_pipeline.py
```

Expected runtime: about 2 minutes on a modern laptop (stage 3 — the 330 rolling XGBoost fits — is the dominant cost).
