# Data Sources

This project relies on three datasets, all obtained via Wharton Research Data Services (WRDS). **None of these are redistributed with this repository** — they are commercial / proprietary data with strict redistribution restrictions. To reproduce the results, you need your own WRDS access.

## Required files

The pipeline (configured in `config.py`) expects the following files at the paths defined in `PATHS`:

| File | Path under data/raw/ | Source | Approx size |
|---|---|---|---|
| CRSP monthly returns | `CRSP_monthly_returns_1995_2024.csv.gz` | WRDS CRSP | ~150 MB |
| Compustat fundamentals | `Compustat_characteristics_1995_2025.csv.gz` | WRDS Compustat | ~25 MB |
| Fama-French factors | `Market_Riskfree.xlsx` | Kenneth French data library | < 100 KB |

The Fama-French file is publicly available; CRSP and Compustat are not.

## How to obtain the data (WRDS users)

### 1. CRSP monthly returns

**Source:** WRDS → CRSP → Stock / Security Files → Monthly Stock File

**Query setup:**
- **Date range:** `1995-01-01` through `2024-12-31`
- **Security identifier:** All US common equities (share codes 10, 11, 12)
- **Output format:** CSV (gzipped)

**Required columns:**

| WRDS column | Used for |
|---|---|
| `PERMNO` | Permanent issue identifier (join key) |
| `date` | Month-end date |
| `RET` | Monthly return (decimal) |
| `PRC` | Closing price |
| `SHROUT` | Shares outstanding (thousands) |
| `CFACSHR` | Cumulative adjustment factor for shares |
| `PRIMEXCH` | Primary exchange code (N/Q/A) |
| `SHRCD` | Share code (10/11/12 for US common equities) |

Note: `RET` occasionally contains string sentinel values ('B', 'C', etc.) which the pipeline coerces to NaN in `ingestion.load_crsp()`.

### 2. Compustat fundamentals

**Source:** WRDS → Compustat / North America → Fundamentals Annual

**Query setup:**
- **Date range:** `1995-01-01` through `2025-12-31` (extra year buffer for lag)
- **Link to CRSP:** Use the WRDS CRSP-Compustat linking table (`ccmxpf_linktable`) to obtain the `LPERMNO` field
- **Output format:** CSV (gzipped)

**Required columns:**

| Compustat column | Concept | Used for |
|---|---|---|
| `LPERMNO` or `PERMNO` | Link to CRSP | Join key |
| `datadate` | Fiscal period end | Lagged by 5 months for availability |
| `ceq` | Common equity (book value) | `b2m` feature |
| `ib` | Income before extraordinary items | `accruals` feature |
| `oancf` | Operating activities net cash flow | `CashFlow2TA`, `CashFlow2Prc`, `accruals` |
| `at` | Total assets | `investment`, `CashFlow2TA`, `accruals` |

If the WRDS export uses `LPERMNO`, `ingestion.load_compustat()` renames it to `PERMNO` automatically.

### 3. Fama-French market and risk-free factors

**Source:** Kenneth French data library — Fama/French 3 Factors monthly file
URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

**Required columns:**

| Column | Description |
|---|---|
| `year` | Calendar year (int) |
| `month` | Calendar month (1..12) |
| `Mkt` | Market return (decimal, monthly) — note: the raw FF file labels this as `Mkt-RF + RF` separately; this project expects them re-aggregated as `Mkt` |
| `RF` | Risk-free rate (decimal, monthly) |

Save as `Market_Riskfree.xlsx` in `data/raw/`. The pipeline reads it with `pandas.read_excel()`.

## License and redistribution

- **CRSP data:** CRSP, LLC. Redistribution prohibited under WRDS terms.
- **Compustat data:** S&P Global Market Intelligence. Redistribution prohibited under WRDS terms.
- **Fama-French factors:** Publicly available from the Kenneth French data library, free for academic and commercial use with attribution.

The `data/raw/` directory is gitignored in this repository. Do not commit raw data files even accidentally. The `.gitignore` already includes `data/raw/*` with an exception only for this README.

## Where the data lives in the pipeline

Once the three files are placed at the paths configured in `config.py`:

1. `src/ingestion.py`:
   - Loads CRSP and Compustat
   - Applies the universe filter (exchanges, share codes, min market cap)
   - Performs the as-of merge with a 5-month Compustat lag and 365-day tolerance

2. `src/feature_engineering.py`:
   - Computes the 10 cross-sectional features from the merged panel
   - Applies cross-sectional Winsorization (1% / 99% per-month)
   - Applies cross-sectional percentile-rank transformation per-month

3. `src/market_model.py`:
   - Loads the Fama-French Mkt/RF file
   - Runs the CAPM regression for each decile + the long-short spread

## Reproducibility notes

- The pipeline uses a fixed random seed (`random_state=0` in `MODEL_PARAMS`). Given identical CRSP/Compustat snapshots and the same WRDS extraction date, results should reproduce to 4+ decimal places.
- CRSP data does occasionally get revised retroactively (historical share counts, etc.). If you pull CRSP at a different point in time than the original April 2026 extraction, you may see slight numerical differences. The methodology and direction of results will not change, only the third or fourth decimal of headline metrics.
- The Compustat fiscal year-end values are static once filed, so Compustat data is more stable across pull dates than CRSP.

## If you don't have WRDS access

You cannot directly run this pipeline without WRDS. However, you can still:

1. **Read the methodology** in `docs/methodology.md` — full technical writeup
2. **Read the report** in `docs/report.md` — published results and academic citations
3. **Inspect the source code** in `src/` — every transformation is documented
4. **Adapt the pipeline** to a different data source — the feature engineering and rolling-window structure are dataset-agnostic, so a similar pipeline could be built on, e.g., daily Yahoo Finance data or Quandl's WIKI prices (with adjusted formulas).

For academic use specifically, your university library may already have WRDS access through a research subscription.
