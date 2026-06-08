# Methodology

This document explains the technical decisions behind the rolling XGBoost cross-sectional return prediction model. It is intended for technical reviewers. The companion `features.md` covers the inputs; this document covers everything else.

> **Numbers in this document are the corrected, leak-free figures** (annualized Sharpe 1.03, monthly alpha +2.19%, alpha t-stat 6.08, beta −0.43). An earlier version reported ~2.5 Sharpe; a look-ahead bias in `vol_12m` was found and fixed. See `docs/leakage_audit.md`.

## 1. Problem Framing

### 1.1 What we are trying to predict

The model predicts a stock's **cross-sectionally demeaned monthly return**:
```
adj_ret_t = RET_t − mean(RET_t across all stocks in month t)
```

We are explicitly not predicting the market's monthly return; that is captured separately in the CAPM regression. The XGBoost model predicts the *relative ranking* — which stocks outperform their peers.

This framing translates to a tradable signal. A long-short portfolio is dollar-neutral within each month; the market-level return component cancels, leaving cross-sectional alpha (and, as it turns out here, a residual negative market beta — see §6.3).

### 1.2 Why cross-section instead of time-series

Two reasons. First, cross-sectional prediction has far more data: ~30 years × ~4,000 stocks ≈ 1.4M observations, vs. ~360 monthly observations for a single time-series. Second, the academic literature on return prediction overwhelmingly favors cross-sectional factor models, and the decile-sort backtest is the standard evaluation tool.

### 1.3 Why XGBoost specifically

A linear cross-sectional regression handles each feature independently and cannot capture interactions (e.g., "high `b2m` matters more for low-`vol_12m` stocks") without explicit terms. XGBoost captures interactions natively via tree splits.

The trade-off is interpretability. We don't extract SHAP or partial-dependence plots; for this project the empirical outcome (corrected Sharpe 1.03, alpha t-stat 6.08) and the small universe of 10 well-understood features keep model decisions in a familiar space.

---

## 2. Data Engineering

### 2.1 The two source datasets

- **CRSP** provides monthly prices, returns, shares outstanding, and exchange/share-code classifications.
- **Compustat** provides fundamentals: book equity, income, cash flow, total assets.

The CRSP-Compustat link table (WRDS) maps each Compustat company to its CRSP `PERMNO`. The pipeline assumes this linking is done in the WRDS export (`PERMNO` or `LPERMNO`).

### 2.2 The universe filter

```
SHRCD ∈ {10, 11, 12}                    # US common equities
PRIMEXCH ∈ {'N', 'Q', 'A'}              # NYSE, NASDAQ, AMEX
marketcap_lag >= 10,000 ($K)             # Min $10M market cap
```

- **Share codes:** excludes preferred stock, ADRs, ETFs, closed-end funds.
- **Exchanges:** excludes Pink Sheets / OTC / de-listed names.
- **Min market cap:** excludes micro-cap shells (a soft $10M floor).

After these filters, ~2.1M (PERMNO, month) observations across 1995–2024.

### 2.3 The as-of merge with 5-month Compustat lag

This is one of the most important leakage-prevention decisions. Compustat fiscal year-end data is not public on fiscal year-end; the 10-K typically follows 60–90 days later, plus vendor processing. The pipeline shifts `datadate` forward 5 months:
```python
cstat["date"] = cstat["datadate"] + DateOffset(months=5)
```
So a December 2010 fiscal year-end becomes usable in May 2011 — conservative. The as-of merge then matches each CRSP month to the most recent Compustat row with `effective_date <= CRSP date`, within a 365-day tolerance (preventing stale data from propagating indefinitely).

### 2.4 What gets dropped

- ~75% of CRSP rows survive the universe filter.
- ~50% of Compustat rows have all required fields populated.
- The as-of merge drops CRSP rows with no Compustat match within tolerance.

End result ~1.56M merged rows; feature engineering drops ~12% (first-year rows where 12-month-lookback features are NaN), and a `yr >= 1995` coverage filter trims a little more, leaving a final modeling panel of ~1.31M rows.

---

## 3. Feature Engineering

Per-feature documentation is in `docs/features.md`. This covers cross-cutting decisions.

### 3.1 Per-month cross-sectional Winsorization

```python
dat.groupby("date")[col].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
```
**Within each month** is the key word: feature distributions shift over time (a "high" `b2m` in 1995 differs from 2020; a "high" `vol_12m` in 2008 looks average in 2017). A pooled Winsorize would use bounds dominated by the most volatile periods.

### 3.2 Percentile rank transformation

```python
dat.groupby("date")[sig].rank(pct=True)
```
This is (1) leakage-safe — rank depends only on the same month's cross-section, not any pooled statistic — and (2) outlier-robust. The actual model inputs are the `_pct_rank` columns.

### 3.3 Why keep the raw feature columns

Both raw and rank columns are kept for diagnostics, audit trail, and sample-size checks. Only `_pct_rank` columns are passed to XGBoost.

---

## 4. Modeling

### 4.1 Why rolling-window training

A single train/test split is inappropriate: market regimes change, a static training set ages badly, and the portfolio-level claim requires performance across the full 25-year period. The rolling window refits every month on the prior ~60 months, simulating real deployment: at month t, train on what's available, predict t+1, observe, repeat.

### 4.2 The "60-month window" that is actually 61 months

```python
train_dat = dat[(dat["month_num"] >= t) & (dat["month_num"] <= t + 60)]
```
Inclusive on both endpoints, so 61 cohorts. Preserved verbatim from the original notebook for numerical reproducibility. Honest interview answer: "the variable says 60 and the report claims 60, but the implementation is 61; both are reasonable and we preserved the original code."

### 4.3 XGBoost hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `reg:squarederror` | Standard regression objective |
| `eval_metric` | `rmse` | Matches the objective |
| `max_depth` | 4 | Shallow trees to limit overfitting on noisy monthly cross-sections |
| `min_child_weight` | 1 | Allows splits on small subgroups |
| `gamma` | 0.2 | Mild pruning |
| `subsample` | 0.8 | Row subsampling per tree |
| `colsample_bytree` | 0.8 | Column subsampling per tree |
| `reg_alpha` | 0 | No L1 |
| `reg_lambda` | 1 | Default L2 |
| `learning_rate` | 0.1 | Standard mid-tier rate |
| `n_estimators` | 40 | Small ensemble — weak per-month signal, overfitting is the bigger risk |
| `tree_method` | `hist` | Histogram-based splits for speed |
| `random_state` | 0 | Reproducibility |

Not heavily tuned; a formal Optuna/grid search is future work. Small trees + small ensemble reflect the low signal-to-noise of monthly cross-sectional prediction.

### 4.4 Regression vs. classification

The model predicts a continuous demeaned return, then sorts into deciles. Classification ("will be in top decile") would discard magnitude information; regression-then-sort consistently outperforms in factor-modeling literature.

### 4.5 Performance characteristics

The 330-iteration rolling loop completes in ~1.5 minutes. Each iteration fits on ~250,000 training rows and predicts on ~3,500 test stocks. Total predictions: ~1,195,000 stock-month observations.

---

## 5. Portfolio Construction

### 5.1 Decile sort

```python
df["rank_order"] = df.groupby(["yr","month"])["predicted_adj_ret"].rank(method="first")
df["rank"] = df.groupby(["yr","month"])["rank_order"].transform(lambda x: pd.qcut(x, 10, labels=False))
```
Decile 0 = lowest 10% predicted; Decile 9 = highest. `method="first"` breaks ties deterministically.

### 5.2 The long-short spread

```
spread_t = decile_9_return_t − decile_0_return_t
```
The classical academic long-short factor portfolio. (Note: in this corrected run the two legs are *not* beta-matched — Decile 0 has higher market beta than Decile 9 — so the spread carries a net negative beta. See §6.3.)

### 5.3 The yr >= 2000 filter

Although the rolling backtest's first prediction is available in July 1997, the portfolio construction filters to `yr >= 2000` for early-year data stability. January 2000 – December 2024 = 300 months ≈ 25 years.

### 5.4 Equal-weight within deciles

```python
meanret = pred_2000.groupby(["yr","month","rank"])["RET"].mean()
```
Equal-weight is the academic standard; value-weight is also defensible. The team chose equal-weight; this rebuild preserves it.

---

## 6. Statistical Inference

### 6.1 Per-decile statistics

| Statistic | Formula |
|---|---|
| Mean monthly return | `RET.mean()` |
| Std of monthly returns | `RET.std()` |
| t-statistic vs zero | `sqrt(n−1) × mean / std` |
| Monthly Sharpe | `mean / std` |
| Annual Sharpe | `sqrt(12) × monthly_sharpe` |

The t-statistic uses (n−1); annualization assumes serially uncorrelated monthly returns.

### 6.2 Why the t-stats matter

The corrected Sharpe of 1.03 is meaningful, but the t-statistic answers "how unlikely is this if the true mean were zero?" For 300 months:
- Monthly mean: 0.0193
- Monthly std: 0.0648
- t-statistic: `sqrt(299) × 0.0193 / 0.0648 ≈ 5.15`

A t-statistic of 5.15 corresponds to a very small p-value (well under 1e-6); the spread return is not plausibly zero. (The original leaked figure was t ≈ 12.6 — inflated by the look-ahead bias.)

### 6.3 The CAPM decomposition

```
excess_returns_p_t = α_p + β_p × mkt_excess_t + ε_p_t
```
For deciles 0..9, `excess_returns = RET − RF`; for the spread, `excess_returns = RET` (self-financing).

The spread's **β = −0.43** says the long-short portfolio is net-*short* the market — a consequence of the short leg (Decile 0) having a higher market beta (≈1.61) than the long leg (Decile 9, ≈1.18). The **α = +2.19%/mo** is what remains after removing that market exposure, and it is strongly significant (t = 6.08). This is the rigorous answer to "is this just market beta?" — no: not only is the residual alpha significant, the net beta is negative, so the alpha is not leveraged long-market exposure.

### 6.4 The decile alpha pattern (corrected)

| Decile | Alpha | t-stat |
|---|---:|---:|
| 0 (lowest) | −1.78% | −4.68 |
| 1 | −0.61% | −2.50 |
| 2 | −0.19% | −1.03 |
| 3 | +0.11% | +0.74 |
| 4 | +0.24% | +1.68 |
| 5 | +0.25% | +1.84 |
| 6 | +0.29% | +2.16 |
| 7 | +0.31% | +2.09 |
| 8 | +0.27% | +1.44 |
| 9 (highest) | +0.41% | +1.51 |

The pattern is broadly monotonic from Decile 0 through the middle deciles, then flattens at the top. The short leg is strongly significant (Decile 0: −1.78%, t = −4.68); the long leg is positive but **not** statistically significant (Decile 9: +0.41%, t = +1.51). So the model identifies "losers" with confidence but "winners" only weakly — the spread's significance comes mainly from the short side. A model with no real signal would show roughly zero alpha at every decile, which is not what we observe on the lower deciles.

---

## 7. What This Methodology Doesn't Address

### 7.1 Transaction costs

The backtest assumes frictionless trading. On a corrected gross monthly return of 1.93%, realistic round-trip costs (a rule-of-thumb 50–150 bps for monthly rebalancing) are a **large** relative drag — on the order of 25–75% of gross — and the high-turnover reversal signal makes this worse. Net-of-cost performance would be materially lower than the gross figures; the gross 1.03 Sharpe should be read as an upper bound.

### 7.2 Short-selling constraints

The spread requires shorting Decile 0 ("losers") — and since the corrected spread is driven mainly by the short leg, this dependence is material. Borrow may be unavailable or expensive for the hardest-to-borrow names (often exactly the most-negative-predicted). A more realistic backtest would constrain the short leg to borrowable names or go long-only.

### 7.3 Capacity constraints

Decile 0 and Decile 9 each hold ~400 stocks/month, so the strategy is nominally scalable, but position limits, small-cap market impact, and sector concentration are not analyzed here.

### 7.4 Hyperparameter tuning

Hyperparameters were chosen by intuition, not formal search. The corrected result is an "out-of-the-box" XGBoost result.

### 7.5 Robustness to alternative universe definitions

Headline numbers depend on the specific universe (SHRCD 10–12, PRIMEXCH N/Q/A, market cap ≥ $10M). No sensitivity analysis is presented.

---

## 8. Summary of Defensible Claims

In order of confidence:

1. **The pipeline produces a statistically significant, leak-free out-of-sample result:** Sharpe 1.03, monthly alpha +2.19%, alpha t-stat 6.08 over 300 months. (The earlier ~2.5 Sharpe was inflated by a look-ahead bias, since corrected — see `docs/leakage_audit.md`.)
2. **The signal is alpha, not market beta.** The spread has β = −0.43 and α = +2.19%/mo (t = 6.08); the alpha survives removal of market exposure, and the net beta is negative.
3. **The short leg is the engine.** Decile 0 alpha = −1.78% (t = −4.68) is the strongest, most significant decile effect; the long leg is positive but not significant. The decile pattern is broadly monotonic on the lower-to-middle deciles.
4. **Look-ahead bias was identified and corrected.** The original `vol_12m` window included the contemporaneous month; it is now lagged to t−1, matching the lag discipline of every other feature. Compustat is lagged 5 months; market-cap inclusion uses lagged values; rolling-window training prevents future leakage. The audit trail is in `docs/leakage_audit.md`.
5. **The factor signals have academic foundations.** Each is grounded in peer-reviewed work (citations in `features.md`); 9 named signals + 1 size control = 10 inputs.

The weaker claims and limitations are documented in §7. Most notably, the corrected result is a frictionless gross result that is highly cost-sensitive and short-leg-dependent — a credible research finding, not a deployable strategy.
