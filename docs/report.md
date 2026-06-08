# Rolling XGBoost Regression Tree Quantitative Model — Final Report

**Institution:** Goizueta Business School, Emory University
**Date:** April 2026 (metrics corrected May 2026 — see note below)

> **Note on numbers in this document.** The April 2026 submitted version reported an annualized Sharpe of ~2.52. A subsequent feature audit (see `docs/leakage_audit.md`) found a look-ahead bias in the idiosyncratic-volatility feature (`vol_12m`): its rolling window included the prediction month's own return. After correcting the leak (lagging the feature by one month), the honest figures are an annualized Sharpe of **1.03**, monthly CAPM alpha of **+2.19%** (t = 6.08), and market beta of **-0.43** over 300 out-of-sample months. All numbers in this document reflect the corrected, leak-free pipeline. Where the original (leaked) figures are referenced, they are clearly labeled.

---

## 1. Introduction

This project constructs a quantitative model that predicts one-month-ahead cross-sectional stock returns using a rolling XGBoost regression tree framework. Monthly predicted returns are used to form decile portfolios, and the performance of the long-minus-short (diff) portfolio is evaluated in terms of risk-adjusted returns and market exposure.

Data come from CRSP (monthly returns) and Compustat (annual accounting variables), covering common domestic equities listed on NYSE, AMEX, and NASDAQ from 1995 to 2024. The rolling backtest produces out-of-sample predictions over 330 months (July 1997 through December 2024); after the year-2000 cohort filter, the portfolio backtest covers January 2000 through December 2024, yielding 300 monthly observations (~25 years).

All features are computed from data available at prediction time. One feature (`vol_12m`) originally contained a look-ahead error — its window included the contemporaneous month — which has since been corrected; the detection and fix are documented in `docs/leakage_audit.md`. After this correction, no feature uses contemporaneous or future information.

---

## 2. Features and Rationale

### 2.1 Existing Features

The model incorporates seven features from the academic literature on cross-sectional return predictability:

- **`ret_2_12`:** Cumulative return from month t−12 to t−2, capturing intermediate-horizon price momentum (Jegadeesh & Titman, 1993).
- **`new_issue`:** 12-month growth in split-adjusted shares outstanding, reflecting share dilution and the new-issue anomaly.
- **`investment`:** Year-over-year growth in total assets, proxying for the conservative investment factor.
- **`accruals`:** Non-cash portion of earnings scaled by total assets, measuring earnings quality.
- **`b2m`:** Book equity divided by market capitalization (value factor).
- **`CashFlow2TA`:** Operating cash flow scaled by total assets.
- **`CashFlow2Prc`:** Operating cash flow scaled by market capitalization.

(With `marketcap` as a size control, these comprise 8 of the 10 model inputs.)

### 2.2 New Features

Two new signals were introduced for this project, both exploiting well-documented behavioral and risk-based return patterns.

**`reversal_1m` (Short-Term Reversal):**
Defined as the negative of last month's return (−RETₜ₋₁). Jegadeesh (1990) and Lehmann (1990) show that stocks with strong positive (negative) returns in the prior month tend to underperform (outperform) the following month, attributed to short-term liquidity effects and microstructure frictions. This signal lets the model exploit short-horizon negative serial correlation, complementing the intermediate-momentum feature `ret_2_12`.

**`vol_12m` (Trailing 12-Month Return Volatility):**
Defined as the rolling standard deviation of monthly returns over the prior 12 months, **ending at month t−1**. The idiosyncratic-volatility puzzle (Ang et al., 2006) documents that high-volatility stocks earn lower subsequent returns. (Note: this feature is where the look-ahead leak occurred; the original implementation's window ended at month t rather than t−1. See §4 and `docs/leakage_audit.md`. A controlled ablation shows that, once correctly lagged, this signal contributes modestly — see §4.)

---

## 3. Model and Implementation

XGBoost sequentially builds an ensemble of shallow regression trees, each correcting the residuals of the preceding ensemble. Regularization (gamma, reg_lambda, min_child_weight) and subsampling (subsample = 0.8, colsample_bytree = 0.8) reduce overfitting and variance. The model uses max_depth = 4, n_estimators = 40, learning_rate = 0.1.

A 60-month rolling window is used: for each prediction month t, the model trains on the prior window of data, then predicts month t. Training and test stocks are restricted to domestic common equities (SHRCD 10–12) on NYSE/AMEX/NASDAQ (PRIMEXCH N/Q/A), with a minimum lagged market capitalization of $10 million.

All features are Winsorized at the 1st and 99th percentiles cross-sectionally each month and converted to percentile ranks (0–1) prior to model fitting.

> **Implementation note:** the parameter is named `rolling_window_months = 60` and the report describes a "60-month rolling window," but the implementation slice is inclusive on both endpoints, so the actual training window contains 61 monthly cohorts. Preserved verbatim for numerical reproducibility.

---

## 4. Portfolio Construction and Performance

Each month, stocks with valid predicted returns are sorted into decile portfolios (Decile 0 = lowest predicted return; Decile 9 = highest). Equal-weighted portfolio returns are computed per decile. The long-minus-short (diff) portfolio goes long Decile 9 and short Decile 0.

The diff portfolio generates an average monthly return of **+1.93%** with an annualized Sharpe ratio of **1.03**, statistically significant (t-stat = **5.15**). A market-model regression of the diff portfolio on the market excess return yields a monthly alpha of **+2.19%** (t-stat = **6.08**), a beta of **−0.43** (t-stat = **−5.55**), and an R² of **0.094**.

> *Pre-correction (leaked) figures, for reference: monthly return +4.79%, Sharpe 2.53, t-stat 12.64, alpha +4.36%, alpha t-stat 13.24, beta +0.73, R² 0.257. These were inflated by the `vol_12m` look-ahead leak documented in `docs/leakage_audit.md`.*

The corrected beta is **negative** (−0.43), meaning the long-short spread is net-short the market rather than net-long: the short leg (Decile 0) carries a higher market beta (≈1.61) than the long leg (Decile 9, ≈1.18), so shorting the losers contributes negative net market exposure. The spread's alpha is therefore not leveraged market beta.

The spread is driven primarily by the **short leg**: Decile 0 has a strongly significant negative alpha (**−1.78%/mo, t = −4.68**), while Decile 9's positive alpha (**+0.41%/mo, t = +1.51**) is *not* statistically significant on its own. The model identifies "losers" with much more confidence than "winners."

---

## 5. Handling Outliers

Extreme values in returns and accounting ratios can distort tree splits and the percentile-rank transformation. Two procedures mitigate this.

First, **monthly cross-sectional Winsorization** clips all features and the target at the 1st and 99th percentiles, capping extremes without dropping observations.

Second, all Winsorized features are converted to **cross-sectional percentile ranks (0–1)** before training, neutralizing any residual outliers because rank is insensitive to magnitude.

---

## 6. Key Takeaways

### Main findings

- The rolling XGBoost model produces statistically significant out-of-sample predictive power: the diff portfolio achieves a monthly alpha of **+2.19%** (t = 6.08) and an annualized Sharpe of **1.03** over the 2000–2024 backtest.
- The two new features (`reversal_1m`, `vol_12m`) are complementary but modest: a controlled ablation (`src/run_ablation.py`) shows they add roughly **+0.09 to the annualized Sharpe (~+10%)** over an 8-feature academic baseline, and are weak in isolation (Sharpe 0.38). Most of the model's signal comes from the academic factor set.
- The spread is **net-short the market** (beta = **−0.43**); it is not beta-neutral.
- The decile alphas are **broadly monotonic** across the lower-to-middle deciles, but the long leg flattens: Decile 0 alpha = **−1.78%** (raw-return t = −1.19, not significant; CAPM alpha t = −4.68); Decile 9 alpha = **+0.41%** (t = +1.51, not significant). The significant short leg drives the spread.
- An initially implausible result (Sharpe ~2.5) prompted a feature audit that found and fixed a look-ahead leak; the corrected figures are reported throughout (see `docs/leakage_audit.md`).

### Limitations and suggested improvements

- **Transaction costs and bid-ask spreads** are not modeled. On a corrected gross monthly return of 1.93%, realistic round-trip costs (50–150 bps) are a *large* relative drag — roughly 25–75% — especially given the high-turnover reversal signal. Net-of-cost performance would be materially lower.
- **Short-leg dependence and short-selling constraints:** the spread relies heavily on shorting Decile 0; borrow availability and fees for those names would reduce realized returns.
- **Market exposure** could be neutralized by beta-hedging the spread.
- **Industry neutralization and size-bucketing** could reduce sector and small-firm effects.
- **Hyperparameter tuning** was not formal; a grid/Optuna search is future work.

---

## Tables

### Table 1. Model Performance (Rolling XGBoost, Out-of-Sample, corrected)

| Backtest Period | Avg Rolling R² | % Positive Months | Portfolio Months | Training Window | Horizon |
|---|---|---|---|---|---|
| Jan 2000 – Dec 2024 | 0.0028 | 56.4% | 300 | 60-month rolling (61 in implementation) | 1 month ahead |

*Universe: NYSE/AMEX/NASDAQ common equities (SHRCD 10–12), market cap ≥ $10M. Avg R² and % positive are across the 330 rolling test months; the portfolio backtest is restricted to the 300 months from 2000 onward. (The earlier leaked run showed Avg R² ≈ 0.008 — also inflated by the look-ahead bias.)*

### Table 2. Decile Portfolio Performance (Jan 2000 – Dec 2024, corrected)

| Decile | Mean Return | Std Dev | t-stat | Mo. Sharpe | Ann. Sharpe | Alpha (mo.) | Alpha t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (Short) | −0.68% | 9.83% | −1.19 | −0.069 | −0.24 | **−1.78%** *** | −4.68 |
| 1 | +0.39% | 7.79% | 0.86 | 0.050 | 0.17 | −0.61% ** | −2.50 |
| 2 | +0.72% | 6.65% | 1.87 | 0.108 | 0.37 | −0.19% | −1.03 |
| 3 | +0.94% | 5.84% | 2.78 | 0.161 | 0.56 | +0.11% | +0.74 |
| 4 | +1.03% | 5.49% | 3.24 | 0.187 | 0.65 | +0.24% | +1.68 |
| 5 | +1.01% | 5.27% | 3.32 | 0.192 | 0.67 | +0.25% | +1.84 |
| 6 | +1.05% | 5.23% | 3.47 | 0.201 | 0.70 | +0.29% ** | +2.16 |
| 7 | +1.08% | 5.40% | 3.45 | 0.199 | 0.69 | +0.31% ** | +2.09 |
| 8 | +1.05% | 5.83% | 3.12 | 0.180 | 0.62 | +0.27% | +1.44 |
| 9 (Long) | +1.25% | 7.08% | 3.06 | 0.177 | 0.61 | +0.41% | +1.51 |
| **Diff (L−S)** | **+1.93%** | **6.48%** | **+5.15** | **0.298** | **1.03** | **+2.19%** *** | **+6.08** |

*Equal-weighted monthly returns over 300 months. The `t-stat` column is the raw-return t vs zero; `Alpha` is the intercept from a market-model regression vs. the Fama-French Mkt excess return. *** p < 0.01, ** p < 0.05. Diff = Decile 9 minus Decile 0. Note Decile 0's raw return is not significant (t = −1.19) but its CAPM alpha is strongly negative (−1.78%, t = −4.68): the short-leg edge is a beta-adjusted effect, since Decile 0 carries a high market beta (≈1.61). Spread market beta = −0.43 (t = −5.55), R² = 0.094.*

### Table 3. Feature Definitions and Formulas

| Feature | Formula | Description |
|---|---|---|
| `ret_2_12` | ∏(1+RETₜ₋ₖ), k=2…12 − 1 | Past 2–12 month momentum |
| `new_issue` | (SHROUT_adj_t − SHROUT_adj_{t−12}) / SHROUT_adj_{t−12}, with extra 1-month lag | Share issuance dilution signal |
| `investment` | AT_t / AT_{t−1} − 1 | Asset growth (conservative investment factor) |
| `accruals` | (IB − OANCF) / AT | Earnings quality: accruals vs. cash flow |
| `b2m` | CEQ / Market Cap | Book-to-market value ratio |
| `CashFlow2TA` | OANCF / AT | Cash flow yield on total assets |
| `CashFlow2Prc` | OANCF / Market Cap | Cash flow yield on market cap |
| `reversal_1m` ★ | −RETₜ₋₁ | NEW: Short-term return reversal |
| `vol_12m` ★ | σ(RET over t−12 … t−1) | NEW: Trailing 12-month return volatility (lagged to t−1; see leakage_audit.md) |
| `marketcap` | SHROUT × |PRC|, lagged 1 month | Implicit size control (10th model input) |

*★ = new features. All features Winsorized at 1%/99% and converted to cross-sectional percentile ranks before training. 9 named factor signals + 1 size control = 10 model inputs; see `docs/features.md`.*

---

## References

- Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *The Journal of Finance, 61*(1), 259–299.
- Banz, R. W. (1981). The relationship between return and market value of common stocks. *Journal of Financial Economics, 9*(1), 3–18.
- Cooper, M. J., Gulen, H., & Schill, M. J. (2008). Asset growth and the cross-section of stock returns. *The Journal of Finance, 63*(4), 1609–1651.
- Daniel, K., Hirshleifer, D., & Sun, L. (2020). Short- and long-horizon behavioral factors. *Review of Financial Studies, 33*(4), 1673–1736.
- Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance, 47*(2), 427–465.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics, 116*(1), 1–22.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *The Journal of Finance, 45*(3), 881–898.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *The Journal of Finance, 48*(1), 65–91.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics, 105*(1), 1–28.
- Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review, 71*(3), 289–315.
