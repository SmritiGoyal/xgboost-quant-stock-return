# Features

This document describes the 10 cross-sectional features used as inputs to the XGBoost model. Each feature is documented with:

- **Formula** — the exact computation
- **Source data** — which CRSP/Compustat columns it draws from
- **Academic citation** — the paper(s) that established the predictive power of this signal
- **Economic interpretation** — what real-world phenomenon it captures
- **Implementation note** — any non-obvious detail about how it's computed in this codebase

The features are listed in the exact order they appear in the model input matrix.

## Feature count: 10 documented = 9 factor signals + 1 size control

The production code passes **10 features** into XGBoost, all documented below: 9 named factor signals plus `marketcap`, which serves as a size control (it is feature #1). The honest framing is: 9 factor signals + 1 size control = 10 model inputs.

The corrected, leak-free headline metrics are an annualized Sharpe of **1.03**, monthly CAPM alpha of **+2.19%**, and alpha t-stat of **6.08**. (An earlier version reported ~2.5 Sharpe; that figure was inflated by a look-ahead bias in `vol_12m`, since corrected — see `docs/leakage_audit.md`.)

---

## 1. `marketcap` — Implicit size control

**Formula:**
```
marketcap_raw_t = SHROUT_t × |PRC_t|
marketcap_t = marketcap_raw lagged by 1 month within each PERMNO
```

**Source data:** CRSP — `SHROUT` (shares outstanding, thousands), `PRC` (closing price, absolute value)

**Academic basis:** Banz (1981); size (SMB) formalized in Fama-French (1993).

**Economic interpretation:** Smaller stocks have historically delivered higher returns, possibly compensating for lower liquidity, less coverage, or higher information uncertainty.

**Implementation note:**
- `PRC` is taken absolute because CRSP encodes bid-ask midpoint quotes as negative.
- The lag (`groupby("PERMNO").shift()`) ensures the inclusion decision for month t uses the prior month's known market cap.
- The universe filter requires marketcap ≥ $10M (thousands of $).

---

## 2. `new_issue` — Net share issuance signal

**Formula:**
```
SHROUT_adj_t = SHROUT_t / CFACSHR_t                    (split-adjusted shares)
raw_t = (SHROUT_adj_t − SHROUT_adj_{t-12}) / SHROUT_adj_{t-12}
new_issue_t = raw_{t-1}                                 (extra 1-month lag)
```

**Source data:** CRSP — `SHROUT`, `CFACSHR`

**Academic citations:** Daniel & Titman (2006); Daniel, Hirshleifer, Sun (2020); Ritter (1991).

**Economic interpretation:** Firms issue when overvalued and repurchase when undervalued; high net issuance predicts lower future returns. Split-adjustment is essential.

**Implementation note:** The original notebook applies an extra `.shift(1)` after the 12-month change, so the value at month t reflects the change from t-13 to t-1 — a one-month conservative buffer.

---

## 3. `investment` — Asset growth (Fama-French 5-factor)

**Formula:**
```
investment_t = (at_t / at_{t-1}) − 1
```

**Source data:** Compustat — `at`

**Academic citations:** Cooper, Gulen, Schill (2008); Fama & French (2015, CMA).

**Economic interpretation:** Aggressive asset growth tends to precede underperformance (overinvestment / agency cost).

**Implementation note:** Computed annually, carried forward via `merge_asof` (backward); the 5-month Compustat lag ensures public availability.

---

## 4. `accruals` — Earnings quality (Sloan accrual)

**Formula:**
```
accruals_t = (ib_t − oancf_t) / at_t
```

**Source data:** Compustat — `ib`, `oancf`, `at`

**Academic citation:** Sloan (1996).

**Economic interpretation:** Accrual-heavy earnings are lower quality than cash-backed earnings; high-accrual firms tend to underperform.

**Implementation note:** Uses the raw signed accrual value (negative accruals are themselves a positive signal).

---

## 5. `b2m` — Book-to-market (value)

**Formula:**
```
b2m_t = ceq_t / marketcap_t
```

**Source data:** Compustat — `ceq`; CRSP — `marketcap_t`

**Academic citation:** Fama & French (1992).

**Economic interpretation:** High book-to-market (value) stocks tend to outperform growth stocks.

**Implementation note:** Mixing lagged Compustat `ceq` (5-month lag) with current-month `marketcap` gives the freshest B/M consistent with no look-ahead.

---

## 6. `ret_2_12` — Intermediate momentum

**Formula:**
```
ret_2_12_t = [Π over i in {2, 3, ..., 12} of (1 + RET_{t-i})] − 1
```

**Source data:** CRSP — `RET`

**Academic citation:** Jegadeesh & Titman (1993).

**Economic interpretation:** Past 2–12 month winners tend to keep outperforming over the next 1–3 months.

**Implementation note (important):** The deliberate skip of month t-1 avoids contamination from the short-term reversal effect (captured separately in feature #9).

---

## 7. `CashFlow2TA` — Operating cash flow profitability

**Formula:**
```
CashFlow2TA_t = oancf_t / at_t
```

**Source data:** Compustat — `oancf`, `at`

**Academic citations:** Novy-Marx (2013); Fama & French (2015, RMW); Hou, Xue, Zhang (2015).

**Economic interpretation:** Higher cash flow per dollar of assets signals quality/profitability and tends to outperform.

**Implementation note:** Uses operating cash flow (more reliably reported 1995–2024 than free cash flow).

---

## 8. `CashFlow2Prc` — Cash flow yield

**Formula:**
```
CashFlow2Prc_t = oancf_t / marketcap_t
```

**Source data:** Compustat — `oancf`; CRSP — `marketcap`

**Academic citation:** Lakonishok, Shleifer, Vishny (1994).

**Economic interpretation:** Cash earnings yield; high values indicate cheapness or distress, low values richness.

**Implementation note:** Unit handling — `oancf` and `marketcap` interpreted as monetary values per share; typical magnitudes ~0.01–0.10.

---

## 9. `reversal_1m` — Short-term reversal

**Formula:**
```
reversal_1m_t = −RET_{t-1}
```

**Source data:** CRSP — `RET`

**Academic citations:** Jegadeesh (1990); Lehmann (1990).

**Economic interpretation:** Last month's strong performers tend to underperform this month (liquidity provision / overreaction correction).

**Implementation note:** The negation converts the signal so that *higher* `reversal_1m` predicts *higher* future return, consistent with the rank-then-decile sort. One of the two newly-proposed features.

---

## 10. `vol_12m` — Idiosyncratic volatility

**Formula (corrected):**
```
vol_12m_t = std(RET_{t-12}, RET_{t-11}, ..., RET_{t-1})
```

i.e., the rolling 12-month standard deviation of returns **ending at month t-1** (information available at the start of the prediction month).

**Source data:** CRSP — `RET`

**Academic citation:** Ang, Hodrick, Xing, Zhang (2006) — the idiosyncratic-volatility puzzle: high-volatility stocks earn lower expected returns.

**Economic interpretation:** High-volatility stocks attract lottery-like demand, driving prices up and expected returns down — opposite to CAPM.

**Implementation note (corrected — important):** The original notebook computed this as
`groupby("PERMNO")["RET"].rolling(12).std()`, a right-aligned window spanning `[t-11, t]` that **included month t's own return** — the same return being predicted. This was a look-ahead leak: it inflated the long-short Sharpe to ~2.5. The fix lags the feature by one month so the window ends at t-1:
```python
merged["vol_12m"] = g["RET"].rolling(12).std().reset_index(level=0, drop=True)
merged["vol_12m"] = merged.groupby("PERMNO")["vol_12m"].shift(1)
```
After the fix, the corrected Sharpe is 1.03. Full detection-and-fix writeup in `docs/leakage_audit.md`. This is the second of the two newly-proposed features; a controlled ablation shows the two proposed signals together add ~+10% to the Sharpe over the academic baseline.

---

## Feature transformations applied before modeling

After raw computation, all 10 features go through two cross-sectional transformations **per-month**, then are used as model inputs.

### Step 1: Winsorization at 1% / 99%

```
feature_winsorized_t = clip(feature_t, lo=quantile_t(0.01), hi=quantile_t(0.99))
```
Per-month rather than pooled, because feature distributions shift across time.

### Step 2: Cross-sectional percentile rank

```
feature_pct_rank_t = rank(feature_t) / N_t
```
Range (0, 1]. This is the actual model input — a leakage-safe scaling that depends only on the same month's cross-section.

### Step 3: Cross-sectional target demean

```
adj_ret_t = RET_t − mean(RET_t across all stocks in month t)
```
Removes the market-level component; the model predicts relative ranking. The market-level return is captured separately by the CAPM regression in `src/market_model.py`.

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
| 10 | `vol_12m` | Idiosyncratic vol | CRSP | t-12 to t-1 (12-mo std, lagged) | Lower vol → higher |

**The 7 from academic literature** are #2–#8 (the report counts 7 academic factors); with `marketcap` (#1) as a size control that is 8 controls/academic inputs. **The 2 newly proposed** are #9 (`reversal_1m`) and #10 (`vol_12m`). Total model inputs: 10.
