# Features

This document describes the 10 cross-sectional features used as inputs to the XGBoost model. Each feature is documented with:

- **Formula** — the exact computation
- **Source data** — which CRSP/Compustat columns it draws from
- **Academic citation** — the paper(s) that established the predictive power of this signal
- **Economic interpretation** — what real-world phenomenon it captures
- **Implementation note** — any non-obvious detail about how it's computed in this codebase

The features are listed in the exact order they appear in the model input matrix.

## Feature count: 9 documented + 1 implicit = 10 total

The team's final report describes **9 features** (7 from academic literature + 2 newly proposed). However, the production code passes **10 features** into XGBoost — the 9 documented features plus `marketcap` itself as an implicit size control.

This rebuild preserves the 10-feature setup because that is what produced the validated headline metrics (Sharpe 2.53, alpha 4.36% monthly, t-stat 13.24). Removing `marketcap` would change the numbers. The honest framing is: 9 named factor signals + 1 size control = 10 model inputs.

---

## 1. `marketcap` — Implicit size control

**Formula:**
```
marketcap_raw_t = SHROUT_t × |PRC_t|
marketcap_t = marketcap_raw lagged by 1 month within each PERMNO
```

**Source data:** CRSP — `SHROUT` (shares outstanding, thousands), `PRC` (closing price, absolute value)

**Academic basis:** Banz (1981) — "The relationship between return and market value of common stocks" — documents the small-firm effect. Subsequent research (Fama-French 1993) formalizes size (SMB) as a factor.

**Economic interpretation:** Smaller stocks have historically delivered higher returns than larger stocks, possibly compensating investors for lower liquidity, less analyst coverage, or higher information uncertainty. Including `marketcap` lets the model use this signal natively.

**Implementation note:**
- `PRC` is taken absolute because CRSP encodes bid-ask midpoint quotes as negative numbers (a CRSP convention).
- The lag (`groupby("PERMNO").shift()`) ensures the inclusion decision for month t uses the prior month's known market cap, not contemporaneous market cap.
- The universe filter requires marketcap ≥ $10M (in thousands of $) to exclude micro-cap shells.

---

## 2. `new_issue` — Net share issuance signal

**Formula:**
```
SHROUT_adj_t = SHROUT_t / CFACSHR_t                    (split-adjusted shares)
raw_t = (SHROUT_adj_t − SHROUT_adj_{t-12}) / SHROUT_adj_{t-12}
new_issue_t = raw_{t-1}                                 (extra 1-month lag)
```

**Source data:** CRSP — `SHROUT`, `CFACSHR`

**Academic citations:**
- Daniel & Titman (2006) — "Market reactions to tangible and intangible information"
- Daniel, Hirshleifer, Sun (2020) — "Short- and long-horizon behavioral factors"
- Ritter (1991) — "The long-run performance of initial public offerings"

**Economic interpretation:** Companies tend to issue shares when their stock is overvalued and repurchase when undervalued. A high net issuance ratio over the past year predicts lower future returns. The split-adjustment (`SHROUT / CFACSHR`) is essential because stock splits trivially change share counts without changing economic substance.

**Implementation note (subtle):** The original notebook applies an **extra `.shift(1)`** after computing the 12-month change, so the published value at month t reflects the share-count change from t-13 to t-1. This is a one-month conservative buffer ensuring the share count is fully reported and any delayed corporate-action data has been incorporated. Removing this extra shift would change the numerical results.

---

## 3. `investment` — Asset growth (Fama-French 5-factor)

**Formula:**
```
investment_t = (at_t / at_{t-1}) − 1
```

where `at_t` is total assets from the most recent Compustat row available for the firm at month t.

**Source data:** Compustat — `at` (total assets)

**Academic citations:**
- Cooper, Gulen, Schill (2008) — "Asset growth and the cross-section of stock returns"
- Fama & French (2015) — "A five-factor asset pricing model" — formalizes CMA (conservative minus aggressive investment) as a factor

**Economic interpretation:** Firms that grow their balance sheet aggressively (high asset growth) tend to underperform subsequently. Conservative-investment firms tend to outperform. This is often interpreted as overinvestment / agency cost: managers expand even when projects have negative NPV.

**Implementation note:** Computed at the Compustat row level (annual), then carried forward to all CRSP months until the next annual Compustat row arrives (via `merge_asof` with backward direction). The 5-month publication lag on the Compustat side ensures the asset growth is publicly known before being used.

---

## 4. `accruals` — Earnings quality (Sloan accrual)

**Formula:**
```
accruals_t = (ib_t − oancf_t) / at_t
```

**Source data:** Compustat — `ib` (income before extraordinary items), `oancf` (operating cash flow), `at` (total assets)

**Academic citation:** Sloan (1996) — "Do stock prices fully reflect information in accruals and cash flows about future earnings?" — the foundational accruals anomaly paper.

**Economic interpretation:** Accruals are the non-cash component of earnings. Earnings that come mostly from accruals (high `ib` − `oancf`) are lower-quality than earnings backed by actual cash flow. Investors don't fully discount accrual-heavy earnings, so high-accrual firms tend to underperform. This is one of the most replicated cross-sectional anomalies in finance.

**Implementation note:** Negative accruals (cash flow exceeds reported income) are themselves a positive signal — they often indicate conservative reporting. The model uses the raw signed accrual value, not its absolute value.

---

## 5. `b2m` — Book-to-market (value)

**Formula:**
```
b2m_t = ceq_t / marketcap_t
```

**Source data:** Compustat — `ceq` (common equity / book equity), CRSP — `marketcap_t`

**Academic citation:** Fama & French (1992) — "The cross-section of expected stock returns" — establishes value (high B/M) as a robust predictor of higher subsequent returns.

**Economic interpretation:** Stocks trading at low prices relative to book value (high B/M) tend to outperform low-B/M (growth) stocks. The classical interpretation is that value firms carry distress risk that gets compensated, though more recent literature (e.g., Asness et al.) argues it reflects mispricing.

**Implementation note:** Mixing Compustat's `ceq` with CRSP's contemporaneous `marketcap` is correct here — the Compustat value comes lagged by 5 months (publication lag), while the market cap is the current month's known value at the time of decision. This gives the freshest book-to-market estimate consistent with no look-ahead.

---

## 6. `ret_2_12` — Intermediate momentum

**Formula:**
```
ret_2_12_t = [Π over i in {2, 3, ..., 12} of (1 + RET_{t-i})] − 1
```

That is, the cumulative return from month t-12 through month t-2 inclusive (skipping t-1).

**Source data:** CRSP — `RET`

**Academic citation:** Jegadeesh & Titman (1993) — "Returns to buying winners and selling losers: implications for stock market efficiency" — the foundational momentum paper.

**Economic interpretation:** Stocks that have outperformed over the past 2-12 months tend to continue outperforming over the next 1-3 months. The momentum anomaly survives across markets, time periods, and asset classes (Asness, Moskowitz, Pedersen 2013).

**Implementation note (important):** The deliberate skip of month t-1 avoids contamination from the short-term reversal effect. Including the prior month would mix two signals with opposite signs: intermediate momentum (positive predictor) and 1-month reversal (negative predictor). The reversal signal is captured separately in feature #9 below.

---

## 7. `CashFlow2TA` — Operating cash flow profitability

**Formula:**
```
CashFlow2TA_t = oancf_t / at_t
```

**Source data:** Compustat — `oancf`, `at`

**Academic citations:**
- Novy-Marx (2013) — "The other side of value: the gross profitability premium"
- Fama & French (2015) — RMW (robust minus weak profitability) factor in the 5-factor model
- Hou, Xue, Zhang (2015) — q-factor model with investment-to-assets and ROE

**Economic interpretation:** Firms that generate more cash flow per dollar of assets are more profitable and tend to outperform. This is a quality / profitability signal — closely related to (but distinct from) standard ROA which uses earnings rather than cash flow.

**Implementation note:** Uses operating cash flow (`oancf`) rather than free cash flow because `oancf` is more reliably reported across the sample period (1995-2024). Free cash flow requires also netting out capex, introducing additional data variability.

---

## 8. `CashFlow2Prc` — Cash flow yield

**Formula:**
```
CashFlow2Prc_t = oancf_t / marketcap_t
```

**Source data:** Compustat — `oancf`, CRSP — `marketcap`

**Academic citation:** Lakonishok, Shleifer, Vishny (1994) — "Contrarian investment, extrapolation, and risk" — establishes cash flow yield as a strong value signal, often outperforming traditional E/P or B/M.

**Economic interpretation:** A "cash earnings yield" — how much cash flow the firm generates per dollar of market value. High values indicate either genuine cheapness or distress; low values indicate richness. The complement to b2m: where b2m uses book equity, this uses cash flow as the "fundamental."

**Implementation note:** Compustat's `oancf` is in millions of dollars (the WRDS export uses raw dollar reporting), CRSP's `marketcap` is in thousands. The ratio is dimensionless after both are interpreted as monetary values per share, but the numerical magnitude (around 0.01-0.10 in typical observations) reflects monthly cash-flow yield rather than annual.

---

## 9. `reversal_1m` — Short-term reversal

**Formula:**
```
reversal_1m_t = −RET_{t-1}
```

**Source data:** CRSP — `RET`

**Academic citations:**
- Jegadeesh (1990) — "Evidence of predictable behavior of security returns"
- Lehmann (1990) — "Fads, martingales, and market efficiency"

**Economic interpretation:** Stocks that did well last month tend to do poorly this month, and vice versa. This is the opposite of intermediate momentum and is usually attributed to (a) liquidity providers being compensated for absorbing large trades, or (b) overreaction to information that gets corrected in subsequent months.

**Implementation note:** The negation (`−RET`) is deliberate — it converts the signal so that *higher* `reversal_1m` predicts *higher* future return. This makes the signal interpretable consistently with the other features under the rank-then-decile sort: stocks with high feature values go to decile 9 (long), low values go to decile 0 (short).

In the team's report, this is one of the **two newly-proposed features** (alongside `vol_12m`) beyond the standard 7-factor literature set.

---

## 10. `vol_12m` — Idiosyncratic volatility

**Formula:**
```
vol_12m_t = std(RET_{t-11}, RET_{t-10}, ..., RET_t)
```

i.e., the rolling 12-month standard deviation of returns ending at month t.

**Source data:** CRSP — `RET`

**Academic citation:** Ang, Hodrick, Xing, Zhang (2006) — "The cross-section of volatility and expected returns" — establishes the **idiosyncratic volatility puzzle**: high-volatility stocks have *lower* expected returns, contradicting classical risk-return tradeoff theory.

**Economic interpretation:** All else equal, high-volatility stocks attract lottery-like demand from retail investors (lottery preference / overconfidence), driving prices up and expected returns down. This is the opposite of what CAPM predicts. The signal is robust across many specifications and time periods.

**Implementation note:** The original notebook computes this with `groupby("PERMNO")["RET"].rolling(12).std().reset_index(level=0, drop=True)` — the `reset_index` is essential because `groupby + rolling` returns a MultiIndex (PERMNO, original_index) that doesn't align with the flat panel index for assignment. This is a pandas quirk preserved verbatim.

In the team's report, this is the **second of two newly-proposed features**.

---

## Feature transformations applied before modeling

After raw computation, all 10 features go through two cross-sectional transformations **per-month**, then are used as model inputs:

### Step 1: Winsorization at 1% / 99%

For each month independently:
```
feature_winsorized_t = clip(feature_t, lo=quantile_t(0.01), hi=quantile_t(0.99))
```

This caps extreme outliers without removing observations. Per-month rather than pooled because feature distributions shift across time (a "high" book-to-market in 1995 is not the same value as a "high" B/M in 2020).

### Step 2: Cross-sectional percentile rank

For each month independently:
```
feature_pct_rank_t = rank(feature_t) / N_t
```

where `N_t` is the number of stocks in month t. Output range is (0, 1].

This is the actual model input. Percentile rank is a leakage-safe scaling: the rank of a value depends only on its position within the same month, not on any future-information statistics like a pooled standard deviation. It also makes the model insensitive to any residual outliers that survived Winsorization.

### Step 3: Cross-sectional target demean

The regression target is:
```
adj_ret_t = RET_t − mean(RET_t across all stocks in month t)
```

This removes the market-level return component. The XGBoost model is therefore predicting **relative ranking**, not absolute return. The market-level return is captured separately by the CAPM regression in `src/market_model.py`.

---

## Summary table

| # | Feature | Type | Source | Lag/lookback | Direction |
|---|---|---|---|---|---|
| 1 | `marketcap` | Size | CRSP | t-1 month | Smaller → higher exp return |
| 2 | `new_issue` | Issuance | CRSP | t-13 to t-1 | Less issuance → higher |
| 3 | `investment` | Annual fundamentals | Compustat (lagged 5mo) | YoY | Less investment → higher |
| 4 | `accruals` | Annual fundamentals | Compustat (lagged 5mo) | Same year | Lower accruals → higher |
| 5 | `b2m` | Value | Compustat + CRSP | Most recent | Higher B/M → higher |
| 6 | `ret_2_12` | Momentum | CRSP | t-12 to t-2 | Higher momentum → higher |
| 7 | `CashFlow2TA` | Profitability | Compustat (lagged 5mo) | Annual | Higher → higher |
| 8 | `CashFlow2Prc` | Value | Compustat + CRSP | Most recent | Higher → higher |
| 9 | `reversal_1m` | Reversal | CRSP | t-1 | Lower last-month → higher next-month |
| 10 | `vol_12m` | Idiosyncratic vol | CRSP | t-11 to t (12-mo std) | Lower vol → higher |

**The 7 from academic literature** are #1, 2, 3, 4, 5, 6, 7 (or 8 — the report counts 7 academic factors plus 2 new; the boundaries between b2m/CashFlow2Prc/CashFlow2TA can be drawn slightly differently). **The 2 newly proposed** are #9 (`reversal_1m`) and #10 (`vol_12m`).