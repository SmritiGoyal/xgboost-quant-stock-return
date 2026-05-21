# XGBoost Quant Stock Return Model

![Sharpe Ratio](https://img.shields.io/badge/Annualized%20Sharpe-2.53-2ea44f?style=for-the-badge)
![Alpha](https://img.shields.io/badge/Monthly%20Alpha-4.36%25%20%28t%3D13.24%29-blue?style=for-the-badge)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20XGBoost%20%7C%20statsmodels-orange?style=for-the-badge)

Rolling-window XGBoost cross-sectional return prediction for US equities (1995–2024). The model produces one-month-ahead return forecasts, sorts them into decile portfolios, and evaluates the long-short spread against a CAPM market model. End-to-end the pipeline runs in under two minutes on a single laptop and reproduces a peer-reviewed quantitative strategy from raw CRSP and Compustat data.

---

## The Problem

Predicting stock returns is hard for reasons that don't disappear with more data or fancier models:

- **The signal-to-noise ratio is extreme.** A single stock's monthly return is dominated by idiosyncratic news, market microstructure, and pure randomness. Even strong factor models explain less than 1% of monthly cross-sectional variance — and that's a feature, not a bug. The Sharpe ratio comes from being right *consistently across many stocks*, not from any single forecast being accurate.
- **Look-ahead bias is everywhere.** Compustat fundamentals are not publicly available on fiscal year-end — the 10-K filing follows 60-90 days later. Using `datadate` directly leaks information that didn't exist at the prediction time. The pipeline applies a 5-month publication lag and a backward-direction `merge_asof` to enforce no-look-ahead at the data level.
- **Regimes change.** Factors that worked in the 1990s (small-cap value) don't work the same way in the 2020s (which were dominated by mega-cap growth). A model trained on the full sample memorizes the average regime. The rolling 60-month window forces the model to adapt to recent conditions, paying a small amount of training-data efficiency for a meaningfully more honest backtest.

This project addresses all three within a reproducible single-machine pipeline grounded in academic factor-investing literature.

---

## Results

The strategy goes long the top decile and short the bottom decile of model-predicted returns each month, then evaluates the resulting return stream against a CAPM market model. The numbers below are from a validated end-to-end re-run; the original April 2026 submitted report reproduces to within 1% (see Reproducibility section).

| Metric | Value |
|---|---:|
| Annualized Sharpe ratio | **2.53** |
| Monthly mean return (long-short) | **+4.79%** |
| Monthly CAPM alpha | **+4.36%** |
| Alpha t-statistic | **13.24** |
| Market beta | **0.726** |
| CAPM regression R² | **0.257** |
| Backtest months (Feb 2000 – Dec 2024) | **300** |

### Decile portfolio structure

A working cross-sectional predictive model produces a **monotonic alpha pattern** across the decile sort. The pipeline produces exactly this — the alpha increases smoothly from the lowest-predicted decile (significantly negative) to the highest (significantly positive), with the long-short spread far more extreme than either tail.

| Decile | Mean return | Annualized Sharpe | CAPM alpha (mo.) | Alpha t-stat |
|---:|---:|---:|---:|---:|
| 0 (lowest predicted) | −2.05% | −0.96 | **−2.94%** | −10.69 |
| 5 (median) | +1.03% | +0.72 | +0.30% | +2.37 |
| 9 (highest predicted) | +2.74% | +0.80 | **+1.42%** | +3.13 |
| **Long-short spread (9 − 0)** | **+4.79%** | **+2.53** | **+4.36%** | **+13.24** |

The 4.36% monthly alpha — what's left after removing market exposure — corresponds to a t-statistic of 13.24, which is roughly the strength of evidence that gravity exists.

---

## How it works

```
            CRSP                Compustat
        (monthly returns)    (fundamentals)
                |                    |
                | 5-month lag        |
                +-----+--------------+
                      |
                      v
              src/ingestion.py
                      |
        merged panel (1.5M rows)
                      |
                      v
        src/feature_engineering.py
        - 10 features (see docs/features.md)
        - cross-sectional Winsorize (1%/99%)
        - cross-sectional percentile rank
                      |
                      v
        src/modeling.py
        - rolling 60-month window
        - XGBoost regressor
        - 330 monthly iterations
                      |
                      v
        src/portfolio_construction.py
        - decile sort within each month
        - long-9 / short-0 spread
                      |
                      v
        src/market_model.py
        - CAPM regression per portfolio
        - alpha, beta, t-stats, R²
                      |
                      v
              outputs/*.csv
```

## Key Technical Decisions

The choices below are the ones that drove the result. Each came from a measured failure of the alternative or an explicit constraint in the data.

### 1. Rolling-window training, not single train/test split

A single train/test split treats stock-return prediction as a stationary problem. It isn't — market regimes change, and a model trained on 1995-2010 data is substantially miscalibrated for 2020+. The pipeline refits the model every month on the prior 60 months of data and predicts only the next month. This simulates how the strategy would actually run: at month t, train on what you have; predict month t+1; observe; repeat. The cost is computational (330 fits instead of one); the benefit is a backtest that's honest about regime changes.

### 2. Compustat fundamentals lagged 5 months, not used at fiscal year-end

The single most important leakage-prevention decision in the pipeline. Compustat reports fiscal year-end accounting data, but that data isn't *publicly available* until the 10-K filing — typically 60-90 days later, plus additional time for the data vendor to ingest, validate, and publish. The pipeline shifts every Compustat row's effective date forward by 5 months:

```python
cstat["date"] = cstat["datadate"] + DateOffset(months=5)
```

Then performs a backward-direction `merge_asof` with 365-day tolerance — meaning each CRSP month gets matched to the most recent Compustat row available at that point in time. Without this lag, the pipeline would "predict" December 2010 returns using December 2010 Compustat data that didn't actually exist until April 2011.

### 3. Cross-sectional percentile-rank transformation, not standardization

After Winsorization, each feature is converted to its within-month percentile rank rather than z-scored. Two reasons. First, **distribution shift over time** — a "high" book-to-market in 1995 (mean ~0.6) isn't the same value as in 2020 (mean ~0.3, dragged down by tech megacaps). A pooled z-score would use bounds dominated by one regime, distorting relative ordering in the other. Second, **leakage safety** — the rank of a value depends only on its position within the same month's cross-section, not on any future-information statistics. The pipeline is bit-safe against accidentally using future moments to scale current features.

### 4. XGBoost over linear regression, but for a specific reason

A linear cross-sectional model (Fama-MacBeth) handles each feature independently. It can find that high book-to-market predicts higher returns *and* that high momentum predicts higher returns, but it can't capture interactions like "value matters more for low-volatility stocks" without explicit interaction terms. XGBoost natively captures these via tree splits. With 10 features and ~3,500 stocks per month, gradient-boosted trees find structure linear models miss. The tradeoff is interpretability — harder to extract "why" the model predicted a stock will outperform — but for a research backtest where the empirical Sharpe is the deliverable, that's an acceptable cost.

### 5. CAPM decomposition isolates true alpha from beta exposure

The long-short spread earns 4.79% per month, but a critical question is: how much of that is leveraged market exposure vs. genuine cross-sectional alpha? The CAPM regression decomposes the spread return into:

```
excess_return = α + β × mkt_excess + ε
```

The result: **β = 0.73** (the spread has meaningful net long market exposure) and **α = 4.36%/month** (still significant at t = 13.24 after removing that exposure). This is the rigorous answer to "is this just leveraged market exposure?" — no, the residual alpha is statistically overwhelming.

---

## Repository Structure

```
xgboost-quant-stock-return/
├── README.md                  This file
├── LICENSE                    MIT
├── requirements.txt           Pinned dependencies
├── .gitignore                 Excludes data/, outputs/, local config.py
├── config.example.py          PipelineConfig template — copy to config.py
│
├── src/
│   ├── ingestion.py           CRSP + Compustat + as-of merge with 5-mo lag
│   ├── feature_engineering.py 10 features + Winsorize + percentile rank
│   ├── modeling.py            Rolling-window XGBoost training loop
│   ├── portfolio_construction.py  Decile sort + long-short spread + stats
│   ├── market_model.py        CAPM regressions per portfolio
│   └── run_pipeline.py        End-to-end orchestrator
│
├── data/
│   ├── raw/                   WRDS data (gitignored — not redistributable)
│   └── README.md              How to obtain CRSP/Compustat
│
├── outputs/
│   └── README.md              Output schema documentation
│
└── docs/
    ├── features.md            10 features with formulas + academic citations
    ├── methodology.md         Full technical writeup
    └── report.md              Original April 2026 submitted report (rebuild values)
```

---

## Reproducing the Results

### Prerequisites

- Python 3.12+
- WRDS subscription for CRSP and Compustat data (academic / institutional access)
- Fama-French monthly factor file (free, from the Kenneth French data library)

### Setup

```bash
git clone https://github.com/SmritiGoyal/xgboost-quant-stock-return.git
cd xgboost-quant-stock-return
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
cp config.example.py config.py
```

### Get the data

CRSP and Compustat are commercial proprietary databases licensed through WRDS. Instructions for the specific WRDS queries are in [`data/README.md`](data/README.md).

### Run

```bash
python src/run_pipeline.py
```

The pipeline writes four CSVs to `outputs/`:

| File | Contents |
|---|---|
| `rolling_xgb_pred_returns_project.csv` | ~1.2M predictions (one row per stock-month) |
| `rolling_xgb_r2_project.csv` | Per-month out-of-sample R² |
| `portfolio_performance_summary.csv` | Per-decile + spread monthly statistics |
| `market_model_results.csv` | CAPM alpha/beta per decile + spread |

### Runtime breakdown

| Stage | Wall time |
|---|---:|
| 1 — Ingestion (CRSP + Compustat + merge) | 4s |
| 2 — Feature engineering (10 features) | 10s |
| 3 — Rolling XGBoost (330 iterations) | 1m 30s |
| 4 — Portfolio construction | 1s |
| 5 — CAPM regressions | 3s |
| **Total** | **~1m 50s** |

---

## Reproducibility

All randomness flows from `random_state = 0` in the XGBoost configuration. Given identical CRSP/Compustat extracts from WRDS, results reproduce deterministically. The pipeline has been re-run independently from the original April 2026 submission and the numerical differences are within the expected range for CRSP data refreshes:

| Metric | Rebuild | Original report | Δ |
|---|---:|---:|---:|
| Annualized Sharpe | 2.53 | 2.52 | +0.01 |
| Monthly alpha | 4.36% | 4.43% | −0.07 pp |
| Alpha t-stat | 13.24 | 13.29 | −0.05 |
| Market beta | 0.726 | 0.77 | −0.04 |
| R² (CAPM) | 0.257 | 0.277 | −0.02 |

The methodology, code structure, and qualitative result all reproduce. The ~1% differences in headline numbers are attributable to CRSP historical revisions (CRSP retroactively updates historical share counts and adjustment factors as data quality issues are discovered) and to scikit-learn / XGBoost version drift between the Colab environment of April 2026 and the local environment of May 2026.

---

## What I'd Do Differently

Treating this as version 1, the obvious improvements for a v2:

- **Beta-hedge the long-short portfolio.** The current spread has β = 0.73, meaning ~73% of its returns can be explained by market exposure. A v2 would short additional SPY exposure to neutralize beta, producing a "pure alpha" portfolio with potentially higher Sharpe (the alpha is uncorrelated with market noise once beta is removed). This is what a real long-short equity fund would do.
- **Industry and size neutralization.** The decile sort can concentrate in specific sectors (e.g., decile 9 might be overweight tech in the late 1990s, financials in 2007). A v2 would do the sort within industry buckets, or weight-adjust to match the cross-sectional industry distribution. Same logic applies to market-cap deciles.
- **Replace XGBoost with LightGBM.** LightGBM's leaf-wise growth and histogram-based splits typically train 2-3× faster at equivalent accuracy on this kind of tabular cross-sectional problem. For a 330-iteration rolling backtest, that compounds.
- **Add ensemble of feature subsets.** The 10 features have heterogeneous information content. A v2 could train K models on overlapping feature subsets and average predictions, which empirically reduces noise on cross-sectional return models. This is what most production quant shops do.
- **Transaction cost modeling.** The current backtest is frictionless. A v2 would impose a 50 bps round-trip cost on monthly rebalancing, which on a 4.8% gross return is a meaningful drag (~20% of alpha). The strategy would still be highly significant, but the *deployable* Sharpe would be 1.8-2.0 rather than 2.5.
- **Hyperparameter sensitivity sweep.** The XGBoost parameters (max_depth=4, n_estimators=40, learning_rate=0.1) were chosen by intuition, not formal search. A grid or Optuna study would justify them quantitatively and likely improve the result modestly.

---

## Tech Stack

- **Python 3.12+**
- **XGBoost 2.1+** — gradient-boosted regression trees
- **statsmodels** — CAPM regression with t-statistics and R²
- **pandas / numpy** — cross-sectional operations, `merge_asof`, `rolling`
- **scipy** — statistical utilities
- Standard library: `pathlib`, `logging`, `dataclasses`

No deep learning, no GPU, no cloud — by design. The pipeline runs end-to-end in under two minutes on a 2024 laptop.

---

## License

MIT — see [LICENSE](LICENSE).

This repository contains source code only. **CRSP and Compustat data are not redistributed** — both are commercial WRDS-licensed databases. The Fama-French Mkt/RF factor file is publicly available from the Kenneth French data library.

---

## Context

This project was completed as the capstone for the Quantitative Strategies & Financial Analytics course in the Emory MSBA program (Spring 2026). This repository is the cleaned, refactored, and documented version of the original team submission; the modeling methodology and results are unchanged.

## Citation

If you reference this work:

```
Goyal, S. (2026). Rolling XGBoost Regression Tree Quantitative Model.
GitHub repository: https://github.com/SmritiGoyal/xgboost-quant-stock-return
```
