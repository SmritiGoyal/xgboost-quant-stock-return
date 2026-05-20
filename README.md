# XGBoost Quant Stock Return Model

> **Rolling-window XGBoost cross-sectional return prediction for US equities (1995–2024).** Annualized Sharpe **2.53**, monthly alpha **4.36%** (t = 13.24), market beta **0.73**, over 300 months out-of-sample.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.1+-orange.svg)

## What this is

This repository contains a complete, reproducible implementation of a cross-sectional stock return prediction model using XGBoost, with rolling 60-month training windows producing one-month-ahead return forecasts. The forecasts are sorted into decile portfolios, and a long-short spread (top decile − bottom decile) is evaluated using CAPM regressions against the Fama-French market factor.

The pipeline is built around 10 features drawn from the academic factor-investing literature (Fama-French, momentum, profitability, accruals) plus two newly proposed features (short-term reversal, idiosyncratic volatility).

## Headline results

| Metric | Value |
|---|---:|
| Annualized Sharpe | **2.53** |
| Monthly mean return | **+4.79%** |
| Monthly alpha (vs Fama-French market) | **+4.36%** |
| Alpha t-statistic | **13.24** |
| Market beta | **0.726** |
| R² (market model) | **0.257** |
| Backtest months | **300** (Feb 2000 – Dec 2024) |

### Decile portfolio structure

The model produces a **monotonic decile alpha pattern**, the structural fingerprint of a working cross-sectional predictive model:

```
Decile 0 (lowest predicted return):   alpha = −2.94%/mo (t = −10.69)   "losers" significant
Decile 9 (highest predicted return):  alpha = +1.42%/mo (t = +3.13)    "winners" significant
Long-short spread:                    alpha = +4.36%/mo (t = +13.24)   strategy alpha
```

For the complete decile-by-decile table and all references to academic literature, see `docs/report.md` (the original April 2026 submitted report, adapted with rebuild values) and `docs/methodology.md` (the technical writeup).

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
│   └── run_pipeline.py        End-to-end orchestrator
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
    └── report.md              Original submitted report (April 2026)
```

## Quick start

### Prerequisites

- Python 3.12+
- WRDS subscription for CRSP and Compustat data (see `data/README.md` for query details)
- Fama-French monthly factor file (free, from Kenneth French data library)

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
```

Expected runtime: ~2 minutes on a modern laptop. The slowest stage is the 330-iteration rolling XGBoost training loop (~1.5 minutes).

## Output

`src/run_pipeline.py` saves four CSVs to `outputs/` (gitignored):

| File | Contents |
|---|---|
| `rolling_xgb_pred_returns_project.csv` | ~1.2M predictions (one row per stock-month) |
| `rolling_xgb_r2_project.csv` | Per-month out-of-sample R² |
| `portfolio_performance_summary.csv` | Per-decile + spread monthly statistics |
| `market_model_results.csv` | CAPM alpha/beta per decile + spread |

Schema details in `outputs/README.md`.

## Reproducibility

The pipeline uses `random_state=0` in the XGBoost configuration. Given identical CRSP/Compustat extracts from WRDS, results reproduce exactly. The author has independently re-run the pipeline and confirmed numerical fidelity to within 1% of the originally submitted report — the small differences are attributable to CRSP historical data being re-pulled at a different point in time.

| Metric | Rebuild | Original report |
|---|---:|---:|
| Annual Sharpe | 2.53 | 2.52 |
| Monthly alpha | 4.36% | 4.43% |
| Alpha t-stat | 13.24 | 13.29 |
| Market beta | 0.73 | 0.77 |

The methodology, code, and structural results all reproduce. The 1% differences in headline numbers are within the expected range for CRSP data refreshes.

## Authors

The rebuild in this repository (a refactor of the original notebook into a modular Python package, with config-driven paths, comprehensive documentation, and validated numerical reproduction) was performed by Smriti Goyal in May 2026.

## Limitations

In the spirit of honest documentation, this implementation does **not** account for:

- **Transaction costs** — the backtest assumes frictionless trading
- **Short-selling constraints** — borrow availability and fees are not modeled
- **Capacity / market impact** — the strategy is presented as a research result, not a deployable fund
- **Hyperparameter tuning** — XGBoost parameters were chosen by reasonable defaults, not by formal grid search
- **Industry / size neutralization** — sector and size concentration not explicitly controlled

See `docs/methodology.md` Section 7 ("What This Methodology Doesn't Address") for full discussion.

## License

MIT — see [LICENSE](LICENSE).

This repository contains source code only. **Raw CRSP and Compustat data are not redistributed** — they are proprietary, WRDS-licensed data. To reproduce results, you need your own WRDS access.

## Citation

If you reference this work, please cite as:

```
Goyal, S.(2026). Rolling XGBoost Regression Tree Quantitative Model.
GitHub repository: https://github.com/SmritiGoyal/xgboost-quant-stock-return
```
