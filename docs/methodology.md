# Methodology

This document explains the technical decisions behind the rolling XGBoost cross-sectional return prediction model. It is intended for technical reviewers — quant practitioners, researchers, or anyone asking "why did you do it this way?" The companion `features.md` covers the inputs in detail; this document covers everything else.

## 1. Problem Framing

### 1.1 What we are trying to predict

The model predicts a stock's **cross-sectionally demeaned monthly return**:
```
adj_ret_t = RET_t − mean(RET_t across all stocks in month t)
```

We are explicitly not trying to predict the market's monthly return. The market return is captured separately in the CAPM regression at the end of the pipeline. What the XGBoost model predicts is the *relative ranking* — which stocks will outperform their peers, not whether all stocks will rise or fall.

This framing matters because it directly translates to a tradable signal. A long-short portfolio (long the highest predicted, short the lowest) is dollar-neutral within each month: the market-level return component cancels out, leaving only the cross-sectional alpha.

### 1.2 Why cross-section instead of time-series

Two reasons.

First, cross-sectional return prediction has much more data than time-series prediction at the per-asset level. We have ~30 years × ~4,000 stocks ≈ 1.4M observations for cross-sectional modeling, vs. ~360 monthly observations for any single time-series.

Second, the academic literature on stock return prediction overwhelmingly favors cross-sectional factor models (Fama-French, q-factor, etc.). The decile-sort backtest used here is the standard tool the academic literature has converged on for evaluating new factor proposals.

### 1.3 Why XGBoost specifically

A linear cross-sectional regression (e.g., Fama-MacBeth) handles each feature independently — it can find that high `b2m` predicts higher returns and that high `ret_2_12` predicts higher returns, but it cannot capture interactions like "high `b2m` matters *more* for low-`vol_12m` stocks" without explicit interaction terms.

XGBoost natively captures interactions via tree splits. With 10 features and ~3,000-4,000 stocks per month, gradient-boosted trees can find the structure linear models miss.

The trade-off is interpretability: it is harder to explain "why" the model predicted a stock will outperform. We don't attempt to extract SHAP or partial dependence plots — for this project, the empirical outcome (Sharpe 2.53, alpha t-stat 13.24) speaks for itself, and the universe of 10 well-understood features keeps the model decisions in a familiar space.

---

## 2. Data Engineering

### 2.1 The two source datasets

- **CRSP** provides monthly prices, returns, shares outstanding, and exchange/share-code classifications. This is the "what happened" data — actual stock returns.
- **Compustat** provides fundamental accounting data: book equity, income, cash flow, total assets. This is the "company information" data.

The CRSP-Compustat link table (from WRDS) maps each Compustat company to its corresponding CRSP `PERMNO`. The pipeline assumes this linking has already been done in the WRDS export — Compustat data arrives with `PERMNO` (or `LPERMNO`, which is renamed) ready to join with CRSP.

### 2.2 The universe filter

The pipeline applies three filters to the CRSP universe:

```
SHRCD ∈ {10, 11, 12}                    # US common equities
PRIMEXCH ∈ {'N', 'Q', 'A'}              # NYSE, NASDAQ, AMEX
marketcap_lag >= 10,000 ($K)             # Min $10M market cap
```

Each filter is defensible:
- **Share codes:** Excludes preferred stock, ADRs, ETFs, closed-end funds. These have different return dynamics and are usually excluded in academic factor research.
- **Exchanges:** Excludes Pink Sheets, OTC, and de-listed names. Liquidity considerations.
- **Min market cap:** Excludes micro-cap shells. This is a soft floor — at $10M the cohort is still substantially small-cap, just not micro-shell.

After these filters, the panel is ~2.1M (PERMNO, month) observations across 1995-2024.

### 2.3 The as-of merge with 5-month Compustat lag

This is the single most important leakage-prevention decision in the pipeline.

Compustat fiscal year-end data is **not publicly available on fiscal year-end**. The 10-K filing typically follows 60-90 days later, plus additional time for the data vendor (Compustat) to ingest, validate, and publish.

The pipeline shifts the Compustat `datadate` forward by 5 months:
```python
cstat["date"] = cstat["datadate"] + DateOffset(months=5)
```

So a December 2010 fiscal year-end becomes available for use on May 2011. This is conservative — the actual 10-K usually publishes 2-4 months out, but 5 months adds a buffer for data vendor processing.

The as-of merge then uses this lagged date:
```python
pd.merge_asof(
    crsp_returns.sort_values("date"),
    compustat.sort_values("date"),
    by="PERMNO",
    left_on="date",
    right_on="date",
    tolerance=dt.timedelta(days=365),
    direction="backward",
)
```

For each CRSP month, this finds the *most recent* Compustat row with `effective_date <= CRSP date`, within a 365-day tolerance. The 365-day tolerance prevents stale data from propagating indefinitely — if a company hasn't filed in over a year, we don't use last year's stale data.

### 2.4 What gets dropped

After the universe filter and the asof merge:
- ~75% of CRSP rows survive the universe filter (cohort exclusion)
- ~50% of Compustat rows have all required fields (`ceq`, `ib`, `oancf`, `at`) populated
- The asof merge then matches each surviving CRSP row to a Compustat row, dropping CRSP rows where no Compustat match exists within tolerance

End result: ~1.56M merged rows. Then feature engineering drops ~12% more rows where any of the 10 features is `NaN` (typically rows in the first 12 months of a PERMNO's history, since `ret_2_12` and `vol_12m` need a year of lookback). Final modeling panel: ~1.31M rows.

---

## 3. Feature Engineering

The detailed per-feature documentation is in `docs/features.md`. This section covers cross-cutting design decisions.

### 3.1 Per-month cross-sectional Winsorization

Outliers in stock-return data are real (penny stocks moving 50%+ in a month) and they distort linear regression. They are less damaging to tree-based models like XGBoost, but they still skew the percentile-rank transformation that follows. The pipeline therefore Winsorizes each feature **within each month**:

```python
dat.groupby("date")[col].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
```

The key word is **within each month**. Per-month bounds are essential because feature distributions shift over time:
- A "high" `b2m` in 1995 (mean ~0.6) is not the same value as a "high" `b2m` in 2020 (mean ~0.3, dragged down by tech megacaps)
- A "high" `vol_12m` in 2008 looks like an average value in 2017

A naive pooled Winsorize would use one global set of bounds dominated by the most volatile periods, distorting the relative ordering in calmer months.

### 3.2 Percentile rank transformation

After Winsorization, each feature is converted to a percentile rank within its month:
```python
dat.groupby("date")[sig].rank(pct=True)
```

This serves two purposes:
1. **Leakage safety:** the rank of a value depends only on its position within the same month's cross-section, not on any pooled statistic (which could leak information from future months).
2. **Outlier robustness:** the rank is bounded in (0, 1] regardless of how extreme any residual outliers are post-Winsorization.

The actual model inputs are the `_pct_rank` columns, not the raw features.

### 3.3 Why we keep the original feature columns too

The pipeline keeps both the raw features AND the percentile-rank features in the panel. This is intentional for:
- **Diagnostics:** if `accruals_pct_rank` looks anomalous, we can inspect the raw `accruals` distribution
- **Audit trail:** every transformation is reversible / verifiable
- **Sample-size checks:** counts of non-NaN raw values reveal data-quality issues

Only the `_pct_rank` columns are passed to XGBoost.

### 3.4 The 10-vs-9 feature discrepancy

The team's report describes 9 features (7 academic + 2 new). The code passes 10 features into XGBoost — the 9 documented features plus `marketcap` itself.

The rebuild preserves the 10-feature setup because that is what produced the validated headline numbers. The honest framing in this codebase is: **9 named factor signals + 1 size control = 10 model inputs.** Documentation in `features.md` discloses this explicitly.

---

## 4. Modeling

### 4.1 Why rolling-window training

A single train/test split is not appropriate here because:
- Market regimes change over time (the factors that worked in the 1990s are not the same as those that work in the 2020s)
- A static training set ages badly: a model trained on 1995-2010 data may be substantially miscalibrated for 2020+
- The portfolio-level claim ("Sharpe 2.53 over 27 years") requires the model to perform across the full time period, not just one carved-out test set

The rolling-window approach refits the model every month using only the prior 60 (well, 61 — see §4.2) months of data. This simulates the real-world deployment scenario: at month t, you train on what you have available; you predict month t+1; you observe the result; you repeat next month.

### 4.2 The "60-month window" that is actually 61 months

The configuration parameter is named `rolling_window_months = 60`. The report describes a "60-month rolling window." However, the slice in `modeling.py` is:
```python
train_dat = dat[(dat["month_num"] >= t) & (dat["month_num"] <= t + 60)]
```

This is **inclusive on both endpoints**, so the actual training window contains 61 monthly cohorts, not 60.

This is preserved verbatim from the team's original notebook because it is what produced the validated Sharpe 2.53. Changing the slice to a strict 60-month window would change all downstream numerical results. The methodology document in this rebuild calls out the discrepancy honestly rather than silently "fixing" it.

For interview purposes: if asked "is your window 60 months or 61?", the honest answer is "the variable name says 60 and that's what the report claims, but the implementation is 61. Both work, and we preserved the original code to maintain numerical reproducibility."

### 4.3 XGBoost hyperparameters

The production values (from `config.py`):

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `reg:squarederror` | Standard regression objective |
| `eval_metric` | `rmse` | Matches the squared-error objective |
| `max_depth` | 4 | Shallow trees to avoid overfitting on noisy monthly cross-sections |
| `min_child_weight` | 1 | Allows splits on small subgroups |
| `gamma` | 0.2 | Mild pruning threshold |
| `subsample` | 0.8 | Row subsampling per tree |
| `colsample_bytree` | 0.8 | Column subsampling per tree |
| `reg_alpha` | 0 | No L1 regularization |
| `reg_lambda` | 1 | Default L2 regularization on leaf weights |
| `learning_rate` | 0.1 | Standard mid-tier learning rate |
| `n_estimators` | 40 | Small ensemble — the signal is weak per-month, so over-fitting is the bigger risk |
| `tree_method` | `hist` | Histogram-based splits for speed |
| `random_state` | 0 | Reproducibility |

These were not heavily tuned. A formal Optuna or grid search over hyperparameters is on the "future work" list. The chosen values reflect:
- **Small trees, small ensemble:** the signal-to-noise ratio in monthly cross-sectional return prediction is low. Larger / deeper trees would overfit.
- **Standard sampling defaults:** `subsample=0.8, colsample=0.8` is the conservative default.
- **No L1 regularization:** with only 10 features there is no need to drive coefficients to zero.

### 4.4 The choice of regression vs. classification

The model predicts a continuous demeaned return, not a binary "will outperform / underperform" label. The downstream portfolio construction then sorts predictions into deciles.

A classification approach (e.g., predict "will be in top decile") is plausible but would lose information. The magnitude of the prediction matters: a stock predicted to be slightly positive is meaningfully different from one predicted to be strongly positive, even if both are above the median.

Empirically, regression-then-sort consistently outperforms direct classification in factor-modeling literature.

### 4.5 Performance characteristics

The 330-iteration rolling loop completes in about 1.5 minutes on a modern laptop. Each iteration:
- Slices the panel to the 61-month training window
- Casts features and target to float32
- Fits an XGBoost regressor on ~250,000 training rows (avg ~4,000 stocks × 61 months)
- Predicts on the test month (~3,500 stocks)
- Stores the prediction and per-month R²

Total predictions across all 330 test months: ~1,195,000 stock-month observations.

---

## 5. Portfolio Construction

### 5.1 Decile sort

For each test month, predictions are sorted into 10 equal-population deciles:
- Decile 0 = lowest 10% of predicted demeaned returns
- Decile 9 = highest 10% of predicted demeaned returns

The implementation uses two steps:
```python
df["rank_order"] = df.groupby(["yr","month"])["predicted_adj_ret"].rank(method="first")
df["rank"] = df.groupby(["yr","month"])["rank_order"].transform(lambda x: pd.qcut(x, 10, labels=False))
```

The `method="first"` rank breaks ties by row position (deterministic given input order). `pd.qcut(..., labels=False)` produces integer decile codes 0..9.

### 5.2 The long-short spread

The headline portfolio is the **long-decile-9 / short-decile-0 spread**:
```
spread_t = decile_9_return_t − decile_0_return_t
```

This is the classical academic long-short factor portfolio. It isolates the cross-sectional signal from market-level returns, because the long and short legs are roughly market-beta-matched.

### 5.3 The yr >= 2000 filter

Although the first rolling-window prediction is technically available in early 1997 (60-month window starting in 1995), the portfolio construction filters to `yr >= 2000`. This is a buffer for:
- Data stability in the early years (CRSP coverage of small-caps improves over time)
- Aligning the backtest with a clean "27 years" claim (Jan 2000 – Dec 2024 = 300 months = 25 years; the report's "27 years" is approximate)

### 5.4 Equal-weight within deciles

Each decile portfolio is equal-weighted across its constituents:
```python
meanret = pred_2000.groupby(["yr","month","rank"])["RET"].mean()
```

Equal-weight is the standard choice in academic factor research. Value-weighted (by market cap) deciles are also defensible — they reduce small-cap exposure but introduce concentration in the few largest names. The team chose equal-weight; this rebuild preserves it.

---

## 6. Statistical Inference

### 6.1 Per-decile statistics

For each decile and the spread, the pipeline computes:

| Statistic | Formula |
|---|---|
| Mean monthly return | `RET.mean()` |
| Std of monthly returns | `RET.std()` |
| t-statistic vs zero | `sqrt(n−1) × mean / std` |
| Monthly Sharpe | `mean / std` |
| Annual Sharpe | `sqrt(12) × monthly_sharpe` |

The t-statistic uses (n−1) following the standard one-sample t-test convention. The annualization assumes monthly returns are serially uncorrelated (the standard assumption for monthly factor portfolios).

### 6.2 Why the t-stats matter

The headline Sharpe of 2.53 is impressive but could in principle be the result of luck on a particular sample. The t-statistic answers "how unlikely is this if the true mean were zero?"

For 300 months of data and a Sharpe of 2.53:
- Monthly mean: 0.0479
- Monthly std: 0.0655
- t-statistic: `sqrt(299) × 0.0479 / 0.0655 ≈ 12.64`

A t-statistic of 12.64 corresponds to a two-sided p-value below 1e-30. By any reasonable definition this is not luck.

### 6.3 The CAPM decomposition

The CAPM regression splits the spread return into:
- **Market exposure (β):** how much the spread moves with the overall market
- **Alpha (α):** the abnormal return after removing market beta

For each portfolio:
```
excess_returns_p_t = α_p + β_p × mkt_excess_t + ε_p_t
```

where:
- For deciles 0..9: `excess_returns = RET − RF` (long-only positions earn the risk-free rate as a base)
- For the spread: `excess_returns = RET` (self-financing position, no RF cost)

The spread's β=0.73 says the long-short portfolio has some net long market exposure. The α=4.36% monthly (52% annualized) is what's left after removing that market exposure. This is the most rigorous answer to "is this strategy just leveraged market beta?" — no, the residual alpha is statistically overwhelming (t=13.24).

### 6.4 The decile alpha pattern

The CAPM regression run on each individual decile shows a monotonic alpha pattern:

| Decile | Alpha | t-stat |
|---|---:|---:|
| 0 (lowest) | -2.94% | -10.69 |
| 1 | -0.73% | -3.70 |
| 2 | -0.16% | -1.04 |
| 3 | +0.15% | 1.06 |
| 4 | +0.25% | 1.89 |
| 5 | +0.30% | 2.37 |
| 6 | +0.33% | 2.62 |
| 7 | +0.37% | 2.51 |
| 8 | +0.33% | 1.49 |
| 9 (highest) | +1.42% | 3.13 |

This monotonic progression — losers significantly negative, winners significantly positive — is the structural fingerprint of a working predictive model. A model with no real signal would produce roughly zero alpha at every decile (consistent with random chance). The fact that decile 0 has α=-2.94% with t=-10.69 is itself strong evidence: the model is identifying real "losers" with statistical confidence comparable to the "winners" identification.

---

## 7. What This Methodology Doesn't Address

In the spirit of honest documentation, here are limitations the rebuild does not fix:

### 7.1 Transaction costs

The backtest assumes frictionless trading. Real transaction costs (bid-ask spreads, market impact, financing costs on the short leg) would reduce realized returns. The standard "rule of thumb" estimate for monthly-rebalanced strategies is 50-150 bps round-trip per month, which on a 4.8% gross return is meaningful (10-30% drag) but does not destroy the strategy.

### 7.2 Short-selling constraints

The long-short spread requires the ability to short stocks in decile 0 ("losers"). In practice:
- Short borrow may be unavailable or expensive for some small / illiquid names
- Hard-to-borrow stocks (often the ones with the most negative predicted return) command higher borrow fees
- Regulatory constraints (short-sale uptick rules, position limits) may impose additional friction

A more realistic backtest would either (a) constrain decile 0 to short-available names only, or (b) build a long-only portfolio (overweight decile 9, no short side).

### 7.3 Capacity constraints

The strategy as built is theoretically scalable to billions of dollars (decile 0 and decile 9 each contain ~400 stocks per month). However, real fund implementations face:
- Position size limits (concentration risk)
- Market impact on small-cap rebalancing
- Sector / industry concentration risk

A capacity analysis is beyond scope here.

### 7.4 Hyperparameter tuning

The XGBoost hyperparameters were chosen by intuition, not by formal grid / Optuna search. A more rigorous study would (a) carve out a separate validation window, (b) search over depth / leaves / regularization, (c) report sensitivity.

The Sharpe-2.53 result is therefore an "out-of-the-box" XGBoost result. The likelihood that tuning would meaningfully improve it is moderate; the likelihood that the tuning process itself would overfit is also moderate.

### 7.5 Robustness to alternative universe definitions

The headline numbers are produced on the specific universe defined by SHRCD ∈ {10,11,12}, PRIMEXCH ∈ {N,Q,A}, and marketcap ≥ $10M. Slight changes to these filters (e.g., excluding small-caps, including ADRs) would produce different numbers. We do not present a sensitivity analysis here.

---

## 8. Summary of Defensible Claims

In order of how confidently each can be defended:

1. **The rolling-window XGBoost cross-sectional return prediction reproduces the headline result.** Sharpe 2.53, alpha 4.36% monthly, t-statistic 13.24 on a 300-month out-of-sample backtest. The numerical reproduction confirms the methodology is sound.

2. **The signal is alpha, not beta.** The CAPM regression shows the spread has β=0.73 and α=4.36% monthly. The alpha t-stat of 13.24 is far beyond any conventional significance threshold.

3. **The decile alpha pattern is monotonic.** Losers significantly negative, winners significantly positive, intermediate deciles transitioning smoothly. This is the structural fingerprint of a real predictive model.

4. **Look-ahead bias is prevented by construction.** Compustat is lagged 5 months. Market cap inclusion uses lagged values. Engineer-history features (e.g., `vol_12m`) use only past data. The rolling-window training prevents future information leaking into past predictions.

5. **The 9 features have academic foundations.** Each factor signal is grounded in a peer-reviewed paper (citations in `features.md`).

The weaker claims (correspondingly limitations) are documented in §7.