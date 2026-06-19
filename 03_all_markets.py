# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display
from pathlib import Path

from fetch_market import load_or_fetch_market
from csad import compute_csad
from absorption_ratio import compute_ar

DATA_DIR = str(Path(__file__).parent / "data" / "raw")
START    = "2015-01-01"
END      = "2026-06-18"
CCK_END  = "2024-12-31"   # CCK analysis fixed to paper sample

MARKETS = {
    "sp500":     "S&P 500",
    "ibovespa":  "Ibovespa",
    "nifty50":   "Nifty 50",
    "ftse100":   "FTSE 100",
    "dax40":     "DAX 40",
    "nikkei225": "Nikkei 225",
}

# %%
# Load all markets — uses cache for sp500, ibovespa, nifty50
prices = {}
for mkt in MARKETS:
    try:
        prices[mkt] = load_or_fetch_market(mkt, start=START, end=END, data_dir=DATA_DIR)
        print(f"  {MARKETS[mkt]:<14s}  {prices[mkt].shape}")
    except Exception as e:
        print(f"  {MARKETS[mkt]:<14s}  ERRO: {e}")

# %%
# Keep only tickers with >= 95% coverage over the full period,
# so late additions (e.g. DAX expanded from 30->40 in Sep 2021) don't
# truncate the effective time frame.
MIN_COVERAGE = 0.95

def filter_coverage(df: pd.DataFrame, min_cov: float = MIN_COVERAGE) -> pd.DataFrame:
    # Must be present from the first observation (no early-period gaps)
    # AND have >=min_cov overall coverage
    first_obs  = df.apply(lambda s: s.first_valid_index())
    start_date = df.index[0]
    present_from_start = first_obs <= start_date + pd.Timedelta(days=5)  # 5-day grace
    full_cov   = df.notna().mean() >= min_cov
    keep       = df.columns[present_from_start & full_cov]
    dropped    = df.shape[1] - len(keep)
    if dropped:
        print(f"    dropped {dropped} tickers (late start or <{min_cov*100:.0f}% coverage)")
    return df[keep]

prices_clean = {mkt: filter_coverage(df) for mkt, df in prices.items()}

# Forward-fill up to 5 days to handle holidays / minor data gaps,
# then compute log returns. dropna(how="all") keeps unbalanced panels.
returns = {}
for mkt, df in prices_clean.items():
    filled = df.ffill(limit=5)          # fills ≤5 consecutive NaN (holidays)
    returns[mkt] = np.log(filled / filled.shift(1)).dropna(how="all")

# %%
# Summary statistics table
rows = []
for mkt, df in prices_clean.items():
    ret  = returns[mkt]
    flat = ret.values.flatten()
    flat = flat[~np.isnan(flat)]
    rows.append({
        "Market":        MARKETS[mkt],
        "Tickers":       df.shape[1],
        "Assets/day":    int(ret.notna().sum(axis=1).median()),
        "Obs (days)":    len(ret),
        "Period":        f"{ret.index[0].strftime('%Y-%m')} / {ret.index[-1].strftime('%Y-%m')}",
        "Mean ret (%)":  round(flat.mean() * 100, 4),
        "Std ret (%)":   round(flat.std()  * 100, 4),
        "NaN (%)":       round(df.isna().mean().mean() * 100, 1),
    })

summary = pd.DataFrame(rows).set_index("Market")
display(summary)

# %%
# CSAD for all markets
csad = {}
for mkt, ret in returns.items():
    csad[mkt] = compute_csad(ret)
    print(f"  {MARKETS[mkt]:<14s}  CSAD mean={csad[mkt]['CSAD'].mean()*100:.3f}%"
          f"  N median={int(csad[mkt]['N'].median())}")

# %%
# PC1-share (first eigenvalue / N) — rolling 63-day window
ar = {}
for mkt, ret in returns.items():
    ar[mkt] = compute_ar(ret, window=63, k=1, data_dir=DATA_DIR, market=mkt)
    print(f"  {MARKETS[mkt]:<14s}  PC1-share mean={ar[mkt]['AR'].mean():.3f}"
          f"  max={ar[mkt]['AR'].max():.3f}")

# %%
# Dual-axis time series: CSAD (left) vs excess lambda1 (right)
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

events = {
    "China\n(Sep/15)":  "2015-09-01",
    "Brexit\n(Jun/16)": "2016-06-23",
    "Volmageddon\n(Feb/18)": "2018-02-05",
    "COVID\n(Mar/20)":  "2020-03-23",
    "Fed\n(Mar/22)":    "2022-03-16",
    "Lib.D\n(Apr/25)":  "2025-04-02",
}

for ax, (mkt, label) in zip(axes, MARKETS.items()):
    if mkt not in csad or mkt not in ar:
        ax.set_visible(False)
        continue

    common = csad[mkt].index.intersection(ar[mkt].index)
    c = csad[mkt].loc[common, "CSAD"] * 100
    e = ar[mkt].loc[common, "excess_lambda1"]

    ax2 = ax.twinx()

    ax.fill_between(common, c, alpha=0.30, color="#1f77b4")
    ax.plot(common, c.rolling(22).mean(), color="#1f77b4",
            linewidth=1.4, label="CSAD 22d MA (%)")

    ax2.plot(common, e, color="#d62728", linewidth=1.0,
             alpha=0.90, label="PC1 / RMT")
    ax2.axhline(e.mean(), color="orange", linewidth=1.0,
                linestyle="--", alpha=0.85)

    for tag, dt in events.items():
        d = pd.Timestamp(dt)
        if common[0] <= d <= common[-1]:
            ax.axvline(d, color="gray", linewidth=0.8, linestyle=":", alpha=0.85)
            ax.text(d, c.max() * 0.88, tag, fontsize=7.5,
                    ha="center", color="gray", va="top")

    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel("CSAD (%)", color="#1f77b4", fontsize=9)
    ax2.set_ylabel("PC1 / RMT", color="#d62728", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.30)

fig.suptitle("CSAD vs PC1-share (excess over RMT bound) — All Markets",
             fontsize=13, y=1.01)
fig.tight_layout()
plt.savefig("all_markets_csad_pc1.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: all_markets_csad_pc1.png")

# %%
# Comparative summary: CSAD + PC1-share statistics per market
rows2 = []
for mkt in MARKETS:
    if mkt not in csad:
        continue
    c = csad[mkt]["CSAD"]
    a = ar[mkt]["AR"] if mkt in ar else pd.Series(dtype=float)
    e = ar[mkt]["excess_lambda1"] if mkt in ar else pd.Series(dtype=float)

    top_dt = e.idxmax() if len(e) else None
    rows2.append({
        "Market":           MARKETS[mkt],
        "CSAD mean (%)":    round(c.mean() * 100, 4),
        "CSAD std (%)":     round(c.std()  * 100, 4),
        "CSAD max (%)":     round(c.max()  * 100, 4),
        "PC1-share mean":   round(a.mean(), 4) if len(a) else None,
        "PC1-share max":    round(a.max(),  4) if len(a) else None,
        "Excess max":       round(e.max(),  2) if len(e) else None,
        "Peak date":        top_dt.strftime("%Y-%m-%d") if top_dt else None,
    })

tbl = pd.DataFrame(rows2).set_index("Market")
display(tbl)

# %%
# CCK symmetric — full sample and sub-sample for all markets
from cck_symmetric import fit_symmetric, _stars

SUB_START = "2019-01-01"
cck_rows = []

for mkt, label in MARKETS.items():
    if mkt not in csad:
        continue
    for sample, sstart, send in [("Full", START, CCK_END), ("Sub", SUB_START, CCK_END)]:
        sub = csad[mkt].loc[sstart:send]
        res = fit_symmetric(sub)
        b2  = res.params["beta2"]
        p2  = res.pvalues["beta2"]
        cck_rows.append({
            "Market":  label,
            "Sample":  sample,
            "n":       len(sub),
            "beta2":   round(b2, 5),
            "p":       round(p2, 4),
            "sig":     _stars(p2) or "ns",
            "R2":      round(res.rsquared, 3),
        })

cck_tbl = pd.DataFrame(cck_rows)
display(cck_tbl.pivot(index="Market", columns="Sample",
                      values=["beta2", "p", "sig", "R2"]))
