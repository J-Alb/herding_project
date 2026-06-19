# Herding Behavior in Global Equity Markets

Empirical pipeline for a doctoral thesis on herding, reflexivity, and price dynamics
in 8 global equity markets (2015–2026), structured as three essays.

**Author:** Giovani Silva — FGV EAESP, DPA Finanças  
**Repo:** https://github.com/J-Alb/herding_project  
**Stack:** Python 3.11+, pandas, yfinance, statsmodels, matplotlib

> **Note:** `data/raw/` (price CSVs) is not tracked in git — it is generated locally
> by running the scripts. First run downloads from Yahoo Finance (~10–30 min per market);
> subsequent runs load from the local cache in seconds.

---

## Quick orientation

The project has two layers:

| Layer | Files | What they do |
|-------|-------|--------------|
| **Library** | `fetch_market.py`, `fetch_controls.py`, `fetch_institutional.py`, `csad.py`, `absorption_ratio.py`, `cck_symmetric.py`, `cck_quantile.py` | Reusable functions. Each can be imported or run standalone. |
| **Analysis scripts** | `00_exemplo_coleta.py`, `01_exemplo_csad.py`, `02_subsample_analysis.py`, `03_all_markets.py` | Cell-by-cell scripts (`.py` with `# %%` markers — open in VS Code with the Jupyter extension or rename to `.ipynb` for Jupyter Lab). |

---

## Requirements

### Python

Python 3.11 or later.

### Install dependencies

```bash
pip install yfinance pandas numpy requests beautifulsoup4 openpyxl \
            fredapi statsmodels matplotlib scikit-learn
```

Or pin versions:

```
yfinance>=0.2.40
pandas>=2.1
numpy>=1.26
requests>=2.31
beautifulsoup4>=4.12
fredapi>=0.5
statsmodels>=0.14
matplotlib>=3.8
openpyxl>=3.1
```

### External API keys

| Service | Required for | How to get |
|---------|-------------|-----------|
| **FRED API key** | `fetch_controls.py` (interest rates, SOFR, T10Y2Y) | Free at https://fred.stlouisfed.org/docs/api/api_key.html |
| Everything else | No account needed | — |

Set the key before running:

```bash
# Linux / Mac
export FRED_API_KEY=your_key_here

# Windows PowerShell
$env:FRED_API_KEY = "your_key_here"
```

### Optional data file

`fetch_controls.py` can load EMBI spreads from a local Excel file:

```
Serie_Historica_Spread_del_EMBI.xlsx   ← place in the project root
```

Source: IADB / JP Morgan historical spread series. If the file is absent the EMBI
step is skipped gracefully and the rest of the pipeline runs normally.

---

## Directory structure

```
herding_project/
│
├── data/
│   └── raw/                    ← CSV caches (auto-created, not in git)
│
├── _nikkei225_tickers.json     ← pre-extracted Nikkei 225 ticker list
├── _fetch_nikkei_tickers.py    ← re-run only if Nikkei list needs updating
│
├── fetch_market.py             ← price data (6 markets working, 2 pending)
├── fetch_controls.py           ← macro controls (VIX, DXY, FRED, GPR, EPU, EMBI)
├── fetch_institutional.py      ← World Bank: WGI, MktCap/GDP, portfolio flows
│
├── csad.py                     ← CSAD computation + multi-market cache
├── absorption_ratio.py         ← rolling PC1-share (first eigenvalue of corr matrix)
├── cck_symmetric.py            ← CCK OLS (symmetric + asymmetric, Newey-West SE)
├── cck_quantile.py             ← CCK quantile regression
│
├── 00_exemplo_coleta.py        ← walkthrough: data collection for all sources
├── 01_exemplo_csad.py          ← walkthrough: CSAD + CCK for SP500 and Ibovespa
├── 02_subsample_analysis.py    ← SP500 full vs sub-sample + dual-axis plot
└── 03_all_markets.py           ← full pipeline: 6 markets, CSAD, PC1-share, CCK
```

---

## Running the code — step by step

### Step 1 — Collect market price data

```python
from fetch_market import load_or_fetch_market

prices = load_or_fetch_market("sp500", start="2015-01-01", data_dir="./data/raw")
```

Supported markets (6 of 8):

| Key | Index | Source |
|-----|-------|--------|
| `sp500` | S&P 500 | Wikipedia + Yahoo Finance |
| `ibovespa` | Ibovespa | B3 official API + Yahoo Finance |
| `nifty50` | Nifty 50 | Wikipedia + Yahoo Finance |
| `ftse100` | FTSE 100 | Wikipedia + Yahoo Finance |
| `dax40` | DAX 40 | Wikipedia + Yahoo Finance |
| `nikkei225` | Nikkei 225 | Pre-built JSON + Yahoo Finance |

IPC México and CSI 300 are not yet implemented (require manual ticker lists).

### Step 2 — Collect macro controls (optional)

```bash
python fetch_controls.py
```

Downloads VIX, DXY (yfinance), DFF/SOFR/SONIA/T10Y2Y (FRED), GPR, and EPU.
Requires `FRED_API_KEY` in environment for FRED series.

### Step 3 — Compute CSAD

```python
import numpy as np
from fetch_market import load_or_fetch_market
from csad import compute_csad

prices = load_or_fetch_market("sp500", start="2015-01-01", data_dir="./data/raw")
returns = np.log(prices / prices.shift(1)).dropna(how="all")
csad = compute_csad(returns)
# columns: CSAD, Rm, abs_Rm, Rm_sq, D_up, D_down, N
```

To compute and cache CSAD for multiple markets at once:

```python
from csad import compute_csad_all
results = compute_csad_all(
    markets=["sp500", "ibovespa", "nifty50"],
    start="2015-01-01",
    data_dir="./data/raw"
)
```

### Step 4 — Compute the Absorption Ratio (PC1-share)

```python
from absorption_ratio import compute_ar

ar = compute_ar(returns, window=63, k=1, data_dir="./data/raw", market="sp500")
# columns: AR, lambda1, lambda1_rmt, excess_lambda1, n_assets
# excess_lambda1 > 1 = co-movement beyond random noise (Marchenko-Pastur bound)
```

**Cache note:** The AR cache file has no date range in the name (`ar_{market}_w63_k1.csv`).
Delete it if you change the date range so it gets recomputed:

```bash
del data\raw\ar_sp500_w63_k1.csv   # Windows
rm data/raw/ar_sp500_w63_k1.csv    # Mac/Linux
```

### Step 5 — CCK herding tests

```python
from cck_symmetric import fit_symmetric, fit_asymmetric

# Symmetric CCK (Chang, Cheng & Khorana 2000)
res = fit_symmetric(csad)      # OLS with Newey-West SE (lags=5)
# beta2 < 0 and significant → herding

# Asymmetric (Tan et al. 2008) — separates up/down market days
res_asym = fit_asymmetric(csad)
```

Quantile regression extension:

```python
from cck_quantile import fit_quantile_cck

qr = fit_quantile_cck(csad, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90])
```

### Step 6 — Full multi-market pipeline

Run `03_all_markets.py` cell by cell. It loads all 6 markets, filters coverage,
computes CSAD and PC1-share, runs CCK, and saves `all_markets_csad_pc1.png`.

**Important:** `03_all_markets.py` does not save CSAD to CSV. Run `compute_csad_all()`
(Step 3) first if you need CSVs for `02_subsample_analysis.py`.

---

## Sample and analysis periods

| Layer | Period |
|-------|--------|
| Price data / plots | 2015-01-01 → present (updated each run) |
| CCK statistical analysis | 2015-01-01 → 2024-12-31 (fixed paper sample) |
| Sub-sample | 2019-01-01 → 2024-12-31 |

The CCK sample is intentionally capped at 2024 so results are stable across runs.
CSAD and PC1-share plots always reflect the latest available data.

---

## Google Colab — what works and what does not

### What works

- All library code (`csad.py`, `cck_symmetric.py`, `cck_quantile.py`, `absorption_ratio.py`) runs without changes.
- The analysis scripts run cell by cell after uploading the `.py` files.

### What needs adjustment

**1. Install dependencies:**
```python
!pip install yfinance fredapi statsmodels beautifulsoup4 openpyxl -q
```

**2. Mount Drive and set paths:**
```python
from google.colab import drive, userdata
import os, sys
drive.mount("/content/drive")
PROJECT = "/content/drive/MyDrive/herding_project"
sys.path.append(PROJECT)
os.chdir(PROJECT)
os.environ["FRED_API_KEY"] = userdata.get("FRED_API_KEY")
DATA_DIR = f"{PROJECT}/data/raw"
```

**3. Data persistence.** Colab sessions reset — save all CSVs to Drive by setting
`data_dir=DATA_DIR` in every call.

**4. Upload `_nikkei225_tickers.json`** alongside the `.py` files for Nikkei 225 support.

### What does NOT work

| Item | Reason |
|------|--------|
| `pdflatex` | LaTeX not installed in Colab. Use Overleaf instead. |
| `_fetch_nikkei_tickers.py` | Wikipedia scraping will timeout. Use the pre-built JSON. |
| Long downloads without Drive | ~20 min S&P 500 download is lost on session reset without Drive. |

---

## Key design decisions

**Correlation matrix, not covariance.** PC1-share uses `.corr()` so each asset is
standardised to unit variance before PCA. This isolates directional co-movement from
differences in volatility level across assets.

**RMT benchmark.** `excess_lambda1 = λ₁ / (1 + √(N/T))²`. Values > 1 indicate
co-movement that exceeds what random noise would produce (Marchenko-Pastur law).

**Coverage filter.** Tickers with < 95% coverage or a late start date are dropped
before computing the correlation matrix. This matters for DAX 40 (expanded from 30
to 40 members in September 2021).

**Forward-fill ≤ 5 days.** Bridges local holidays without propagating stale prices
across longer gaps.

**Newey-West SE, lags = 5.** HAC standard errors with one trading week of lags
correct for autocorrelation in CSAD.

**Cache by filename.** Every fetch function checks for a CSV before downloading.
Pass `force=True` to bypass.

---

## Markets pending

| Market | Status | What is needed |
|--------|--------|---------------|
| IPC México | Not started | Manual list of ~35 BMV tickers |
| CSI 300 | Not started | Manual list of 300 Shanghai/Shenzhen tickers |
