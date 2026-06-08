# XGBoost Quant Stock Return Model

> **Rolling-window XGBoost cross-sectional return prediction for US equities (1995-2024).** Out-of-sample annualized Sharpe **1.03**, monthly CAPM alpha **+2.19%** (t = 6.08), market beta **-0.43**, over 300 months (Jan 2000 - Dec 2024).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.1+-orange.svg)

## What this is

This repository contains a complete, reproducible implementation of a cross-sectional stock return prediction model using XGBoost, with rolling 60-month training windows producing one-month-ahead return forecasts. The forecasts are sorted into decile portfolios, and a long-short spread (top decile minus bottom decile) is evaluated using CAPM regressions against the Fama-French market factor.

The model uses **10 features**: 8 drawn from the academic factor-investing literature (Fama-French value/investment, momentum, profitability, accruals) and market-cap as a size control, plus **2 self-proposed signals** (short-term reversal, idiosyncratic volatility). See `docs/features.md` for formulas and citations.

> **Note on a corrected look-ahead leak.** An earlier version of this model reported an annualized Sharpe of ~2.5. A feature audit (documented in `docs/leakage_audit.md`) found that the idiosyncratic-volatility feature was computed over a window that included the prediction month's own return — a look-ahead leak. Lagging the feature to information available at prediction time reduces the Sharpe to the honest **1.03** reported here. All headline numbers in this README reflect the corrected, leak-free pipeline.

## Headline results (leak-free)

| Metric | Value |
|---|---:|
| Annualized Sharpe | **1.03** |
| Monthly mean spread return | **+1.93%** |
| Monthly Sharpe | **0.30** |
| Spread t-statistic (vs zero) | **5.15** |
| Monthly CAPM alpha (vs Fama-French market) | **+2.19%** |
| Alpha t-statistic | **6.08** |
| Market beta | **-0.43** |
| R-squared (market model) | **0.094** |
| Backtest months | **300** (Jan 2000 - Dec 2024, ~25 years) |

### Decile portfolio structure

The model produces a broadly monotonic decile-alpha pattern, the structural signature of a working cross-sectional predictive model. The spread is driven primarily by the short leg (decile 0 has a large negative alpha); the long leg (decile 9) is positive but not statistically significant on its own.

```
Decile 0 (lowest predicted return):   alpha = -1.78%/mo (t = -4.68)   "losers" significant
Decile 9 (highest predicted return):  alpha = +0.41%/mo (t = +1.51)   "winners" not significant
Long-short spread:                    alpha = +2.19%/mo (t = +6.08)   strategy alpha
```

For the complete decile-by-decile table and references to the academic literature, see `docs/report.md` and `docs/methodology.md`.

### Feature ablation: contribution of the proposed signals

To isolate how much the two self-proposed signals add over the academic-factor baseline, the pipeline was run on three feature sets holding everything else constant (`src/run_ablation.py`):

| Feature set | Features | Ann. Sharpe | Monthly alpha | Alpha t-stat |
|---|---:|---:|---:|---:|
| Baseline (academic + controls) | 8 | 0.94 | +2.14% | 5.65 |
| Full model (+ proposed signals) | 10 | **1.03** | +2.19% | 6.08 |
| Proposed signals only | 2 | 0.38 | +0.92% | 3.05 |

The proposed signals add roughly +0.09 to the annualized Sharpe (about +10%) over the academic baseline. They are complementary rather than dominant: weak in isolation, modestly additive in combination. This ablation is also what surfaced the volatility-feature leak (see below).

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
        merged panel (1.56M rows)
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
        - alpha, beta, t-stats, R-squared
                      |
                      v
              outputs/*.csv
```

## Repository structure

```
xgboost-quant-stock-return/
├── README.md                  This file
├── LICENSE                    MIT
├── requirements.txt
├── .gitignore
├── config.example.py          Copy to config.py and fill in local paths
│
├── src/
│   ├── ingestion.py           CRSP + Compustat + as-of merge with 5-mo lag
│   ├── feature_engineering.py 10 features + Winsorize + percentile rank
│   ├── modeling.py            Rolling-window XGBoost training loop
│   ├── portfolio_construction.py  Decile sort + long-short spread + stats
│   ├── market_model.py        CAPM regressions per portfolio
│   ├── run_pipeline.py        End-to-end orchestrator
│   └── run_ablation.py        Feature-ablation harness (baseline vs full)
│
├── data/
│   ├── raw/                   WRDS data (gitignored — not redistributable)
│   └── README.md              How to obtain CRSP/Compustat from WRDS
│
├── outputs/
│   └── README.md              Pipeline output schema documentation
│
└── docs/
    ├── features.md            10 features with formulas + citations
    ├── methodology.md         Full technical writeup of methodology
    ├── leakage_audit.md       The volatility-feature leak: detection + fix
    └── report.md              Original submitted report (April 2026)
```

## Quick start

### Prerequisites

- Python 3.12+
- WRDS subscription for CRSP and Compustat data (see `data/README.md` for query details)
- Fama-French monthly factor file (free, from the Kenneth French data library)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/SmritiGoyal/xgboost-quant-stock-return.git
cd xgboost-quant-stock-return

# 2. Set up a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.\.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create local config
cp config.example.py config.py
# Edit config.py with your local paths

# 5. Download data per data/README.md, then place files in data/raw/

# 6. Run the pipeline
python src/run_pipeline.py

# (optional) Run the feature ablation
python src/run_ablation.py
```

Expected runtime: ~2 minutes on a modern laptop for the main pipeline. The slowest stage is the 330-iteration rolling XGBoost training loop.

## Output

`src/run_pipeline.py` saves four CSVs to `outputs/` (gitignored):

| File | Contents |
|---|---|
| `rolling_xgb_pred_returns_project.csv` | ~1.2M predictions (one row per stock-month) |
| `rolling_xgb_r2_project.csv` | Per-month out-of-sample R-squared |
| `portfolio_performance_summary.csv` | Per-decile + spread monthly statistics |
| `market_model_results.csv` | CAPM alpha/beta per decile + spread |

Schema details in `outputs/README.md`.

## Reproducibility and the corrected leak

The pipeline uses `random_state=0` in the XGBoost configuration. Given identical CRSP/Compustat extracts from WRDS, results reproduce exactly.

The original April 2026 report reported an annualized Sharpe of 2.52. This repository's rebuild faithfully reproduced that number (2.53) when run with the original feature code — confirming the refactor preserved the methodology. However, a subsequent feature audit found that the reproduced result, and therefore the original, contained a look-ahead bias in the idiosyncratic-volatility feature: its rolling window included the prediction month's own return. Correcting the leak (lagging the feature by one month) yields the honest figures reported throughout this README.

| Metric | Original report | Rebuild (as-submitted, with leak) | Corrected (leak-free) |
|---|---:|---:|---:|
| Annual Sharpe | 2.52 | 2.53 | **1.03** |
| Monthly alpha | 4.43% | 4.36% | **2.19%** |
| Alpha t-stat | 13.29 | 13.24 | **6.08** |
| Market beta | 0.77 | 0.73 | **-0.43** |

The full detection-and-fix writeup, including the ablation that surfaced it, is in `docs/leakage_audit.md`. The corrected figures are reported here in the interest of honest documentation; a result that initially looks too strong is worth interrogating before it is trusted.

## Limitations

In the spirit of honest documentation, this implementation does **not** account for:

- **Transaction costs** — the backtest assumes frictionless trading. The short-term reversal signal is high-turnover, so net-of-cost performance would be materially lower than the gross figures above.
- **Short-selling constraints** — borrow availability and fees are not modeled. The spread relies heavily on the short leg.
- **Market beta** — the corrected long-short spread carries a negative market beta (-0.43), i.e. it is implicitly net-short the market; it is not beta-neutral.
- **Capacity / market impact** — the strategy is presented as a research result, not a deployable fund.
- **Hyperparameter tuning** — XGBoost parameters were chosen by reasonable defaults, not by formal grid search.
- **Industry / size neutralization** — sector and size concentration not explicitly controlled.

See `docs/methodology.md` Section 7 for full discussion.

## License

MIT — see [LICENSE](LICENSE).

This repository contains source code only. **Raw CRSP and Compustat data are not redistributed** — they are proprietary, WRDS-licensed data. To reproduce results, you need your own WRDS access.

## Citation

```
Goyal, S. (2026). Rolling XGBoost Regression Tree Quantitative Model.
GitHub repository: https://github.com/SmritiGoyal/xgboost-quant-stock-return
```
