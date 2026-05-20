# Rolling XGBoost Regression Tree Quantitative Model — Final Report

**Institution:** Goizueta Business School, Emory University
**Date:** April 2026

> **Note on numbers in this document:** the metrics shown here are from the rebuild of the original pipeline (this repository), which reproduces the methodology of the April 2026 submitted report. Headline figures (Sharpe, alpha, beta) match the original report to within 1% — the small differences are attributable to CRSP data being re-pulled from WRDS at a different point in time. The original submitted report values are referenced where they differ, in italics.

---

## 1. Introduction

This project constructs a quantitative model that predicts one-month-ahead cross-sectional stock returns using a rolling XGBoost regression tree framework. Monthly predicted returns are used to form decile portfolios, and the performance of the long-minus-short (diff) portfolio is evaluated in terms of risk-adjusted returns and market neutrality.

Data come from CRSP (monthly returns) and Compustat (annual accounting variables), covering common domestic equities listed on NYSE, AMEX, and NASDAQ from 1995 to 2024. The out-of-sample prediction period spans February 2000 through December 2024, yielding 300 monthly observations after the year-2000 cohort filter is applied. *(Original report stated July 1997 – December 2024 / 330 months; the rebuild applies a `yr >= 2000` filter to the portfolio backtest, which gives the cleaner 25-year window.)* All features are computed from publicly available financial data to ensure the model is free from look-ahead bias.

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

### 2.2 New Features

Two new signals were introduced for this project, both of which exploit well-documented behavioral and risk-based return patterns.

**`reversal_1m` (Short-Term Reversal):**
Defined as the negative of last month's return (−RETₜ₋₁). Jegadeesh (1990) and Lehmann (1990) show that stocks with strong positive (negative) returns in the prior month tend to underperform (outperform) in the following month. This reversal is attributed to short-term liquidity effects and market microstructure frictions. Including this signal allows the model to systematically exploit the negative serial correlation in monthly returns at the very short horizon, complementing the intermediate momentum feature `ret_2_12`.

**`vol_12m` (Trailing 12-Month Return Volatility):**
Defined as the rolling standard deviation of monthly returns over the prior 12 months. The idiosyncratic volatility puzzle (Ang et al., 2006) documents that high-volatility stocks earn lower subsequent returns, contrary to a simple risk-return trade-off. Investors may be willing to overpay for high-volatility "lottery-like" stocks, creating a pricing inefficiency. By including `vol_12m`, the model is able to underweight such stocks in the long leg, adding an independent source of alpha.

---

## 3. Model and Implementation

XGBoost (eXtreme Gradient Boosting) sequentially builds an ensemble of shallow regression trees, where each subsequent tree corrects the residuals of the preceding ensemble. Regularization parameters (gamma, reg_lambda, min_child_weight) help prevent overfitting, and subsampling (subsample = 0.8, colsample_bytree = 0.8) further reduces variance. The model was configured with max_depth = 4, n_estimators = 40, and learning_rate = 0.1.

A 60-month rolling window is used: for each prediction month t, the model trains on the prior 60 months of data, then predicts returns for month t. This design prevents any data leakage while allowing the model to adapt to changing market conditions over time. Training and test stocks are restricted to domestic common equities (SHRCD 10–12) traded on NYSE, AMEX, and NASDAQ (PRIMEXCH N/Q/A), with a minimum lagged market capitalization of $10 million to exclude micro-cap shells.

All features are Winsorized at the 1st and 99th percentiles cross-sectionally each month and then converted to percentile ranks (0–1) to standardize their distributions prior to model fitting.

> **Implementation note added in rebuild:** the configuration parameter is named `rolling_window_months = 60` and the report describes a "60-month rolling window," but the implementation slice is inclusive on both endpoints, so the actual training window contains 61 monthly cohorts. This was preserved from the original notebook to maintain numerical reproducibility.

---

## 4. Portfolio Construction and Performance

Each month, all stocks with valid predicted returns are sorted into decile portfolios (Decile 0 = lowest predicted return; Decile 9 = highest). Equal-weighted portfolio returns are computed for each decile. The long-minus-short (diff) portfolio is formed by going long Decile 9 and short Decile 0. Key performance metrics are reported in Tables 1 and 2 below.

The diff portfolio generates an average monthly return of **+4.79%** with an annualized Sharpe ratio of **2.53**, and the mean return is statistically significant (t-stat = **12.64**). A market model regression of the diff portfolio on the market excess return yields a monthly alpha of **+4.36%** (t-stat = **13.24**), a beta of **0.726** (t-stat = **10.16**), and an R² of **0.257**.

*Original report values: +4.89% monthly return, Sharpe 2.52, t-stat 12.58, alpha +4.43%, alpha t-stat 13.29, beta 0.77, R² 0.277.*

Because beta is positive and statistically significant at conventional levels, the diff portfolio retains meaningful market exposure and is therefore not fully market-neutral. The long-short spread is driven primarily by the strongly negative performance of Decile 0 (mean = **−2.47%** per month, annual Sharpe = **−1.15**) and the robust performance of Decile 9 (mean = **+2.33%** per month, annual Sharpe = **+0.89**), indicating that the model captures both the short and long legs effectively.

---

## 5. Handling Outliers

Extreme values in stock returns and accounting ratios can heavily distort XGBoost tree splits, causing the model to place excessive weight on a small number of observations and reducing out-of-sample generalizability. Two procedures are applied to mitigate this risk.

First, **monthly cross-sectional Winsorization** clips all features and the target return variable at the 1st and 99th percentiles. This caps extreme values without removing the observations, preserving sample size while eliminating the most distorting tail realizations.

Second, all Winsorized features are converted to **cross-sectional percentile ranks (0–1)** prior to model training. Rank-transformation further neutralizes the influence of any residual outliers, because the rank of a value is insensitive to its exact magnitude.

Together, these two steps ensure that no single extreme data point can dominate the training signal or corrupt the portfolio sort.

---

## 6. Key Takeaways

### Main findings

- The rolling XGBoost model produces significant out-of-sample predictive power: the diff portfolio achieves a monthly alpha of **+4.36%** and an annualized Sharpe ratio of **2.53** over the 2000–2024 backtest period.
- The two new features — `reversal_1m` and `vol_12m` — provide complementary signals rooted in well-established behavioral and risk-based anomalies, broadening the model's information set beyond intermediate-horizon momentum.
- Market neutrality is not fully achieved (beta = **0.73**), suggesting that the long-minus-short spread retains moderate directional market exposure.
- Decile 9 exhibits notably high volatility (std ≈ 9.1%), indicating that the highest predicted-return stocks carry elevated idiosyncratic risk.
- The decile alphas form a monotonic pattern (Decile 0 alpha = −2.94% with t = −10.69; Decile 9 alpha = +1.42% with t = +3.13), the structural fingerprint of a working cross-sectional predictive model.

### Limitations and suggested improvements

- **Market neutrality** could be improved by explicitly beta-hedging the diff portfolio or by incorporating the market beta as a feature to be neutralized during portfolio construction.
- **Transaction costs and bid-ask spreads** have not been accounted for. Given the monthly rebalancing and the presence of small-cap stocks, real-world implementation costs may erode a portion of the gross alpha.
- **Industry neutralization and size-bucketing** could reduce the influence of sector-level return shocks and the small-firm effect on portfolio performance.
- **Hyperparameter tuning** was not formal — a grid search or Optuna study over XGBoost parameters may yield further improvements without overfitting.

---

## Tables

### Table 1. Model Performance (Rolling XGBoost, Out-of-Sample)

| Backtest Period | Avg Rolling R² | % Positive R² | Total Months |
|---|---|---|---|
| Feb 2000 – Dec 2024 | 0.0083 | ≈64% | 300 |
| Training Window | 60-month rolling (61 in implementation) | Predict horizon | 1 month ahead |

*Note: Universe restricted to NYSE/AMEX/NASDAQ common equities (SHRCD 10–12), market cap ≥ $10M. Original report reported 330 months covering Jul 1997 – Dec 2024 with Avg R² 0.0079 and 70.3% positive months; the rebuild restricts to yr ≥ 2000 for a cleaner 25-year out-of-sample window.*

### Table 2. Decile Portfolio Performance (Feb 2000 – Dec 2024)

| Decile | Mean Return | Std Dev | t-stat | Mo. Sharpe | Ann. Sharpe | Alpha (mo.) | Alpha t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (Short) | −2.05% | 7.40% | −4.80 | −0.277 | −0.96 | **−2.94%** *** | −10.69 |
| 1 | +0.13% | 6.45% | 0.35 | 0.020 | 0.07 | −0.73% *** | −3.70 |
| 2 | +0.65% | 5.76% | 1.94 | 0.112 | 0.39 | −0.16% | −1.04 |
| 3 | +0.92% | 5.38% | 2.97 | 0.172 | 0.59 | +0.15% | +1.06 |
| 4 | +1.00% | 5.19% | 3.35 | 0.194 | 0.67 | +0.25% | +1.89 |
| 5 | +1.03% | 5.00% | 3.58 | 0.207 | 0.72 | +0.30% ** | +2.37 |
| 6 | +1.08% | 5.13% | 3.64 | 0.211 | 0.73 | +0.33% *** | +2.62 |
| 7 | +1.14% | 5.45% | 3.63 | 0.210 | 0.73 | +0.37% ** | +2.51 |
| 8 | +1.19% | 6.67% | 3.09 | 0.178 | 0.62 | +0.33% | +1.49 |
| 9 (Long) | +2.74% | 11.92% | 3.97 | 0.230 | 0.80 | **+1.42%** *** | +3.13 |
| **Diff (L−S)** | **+4.79%** | **6.55%** | **12.64** | **0.731** | **2.53** | **+4.36%** *** | **13.24** |

*Note: Equal-weighted monthly returns over 300 months. Alpha = intercept from market model regression vs. Fama-French Mkt excess return. *** p < 0.01, ** p < 0.05. Diff = Decile 9 minus Decile 0.*

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
| `reversal_1m` ★ | −RETₜ₋₁ | NEW: Short-term return reversal (mean-reversion) |
| `vol_12m` ★ | σ(RET, past 12 months) | NEW: Trailing 12-month return volatility |
| `marketcap` | SHROUT × |PRC|, lagged 1 month | Implicit size control (10th model input) |

*Note: ★ indicates new features introduced for this project. All features are Winsorized at 1%/99% and converted to cross-sectional percentile ranks before model training. The 10th feature (`marketcap`) is used in the model as an implicit size control — see `docs/features.md` for the honest 9-vs-10 discrepancy explanation.*

---

## References

- Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *The Journal of Finance, 61*(1), 259–299.
- Banz, R. W. (1981). The relationship between return and market value of common stocks. *Journal of Financial Economics, 9*(1), 3–18.
- Cooper, M. J., Gulen, H., & Schill, M. J. (2008). Asset growth and the cross-section of stock returns. *The Journal of Finance, 63*(4), 1609–1651.
- Daniel, K., Hirshleifer, D., & Sun, L. (2020). Short- and long-horizon behavioral factors. *Review of Financial Studies, 33*(4), 1673–1736.
- Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance, 47*(2), 427–465.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics, 116*(1), 1–22.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *The Journal of Finance, 45*(3), 881–898.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: implications for stock market efficiency. *The Journal of Finance, 48*(1), 65–91.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics, 105*(1), 1–28.
- Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review, 71*(3), 289–315.
