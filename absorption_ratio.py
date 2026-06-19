"""
absorption_ratio.py
-------------------
Rolling Absorption Ratio (AR) as a herding/systemic risk measure.

Motivation
----------
The Absorption Ratio (Kritzman, Li, Page & Rigobon, 2011) measures
the fraction of total variance explained by the first k eigenvectors
of the rolling asset correlation matrix.

    AR_k(t) = sum(lambda_1 ... lambda_k) / sum(all lambda_i)

As assets move together (herding), the correlation matrix is dominated
by one large eigenvalue -- the "market direction" eigenvector. The matrix
approaches singularity and AR -> 1.

This overcomes the main CCK limitation: during COVID, correlations spike
and AR fires correctly, whereas CSAD explodes and CCK reads anti-herding.

Random Matrix Theory (RMT) benchmark
-------------------------------------
For a random (noise-only) correlation matrix with N assets and T observations,
the Marchenko-Pastur law predicts the bulk eigenvalue upper bound:

    lambda_RMT = (1 + sqrt(N/T))^2

Eigenvalues above this bound represent genuine co-movement beyond noise.
The "excess first eigenvalue" lambda_1 / lambda_RMT is a clean herding signal.

Note on N > T
-------------
For the S&P 500 (~490 assets) with short windows (63 days), N > T.
The correlation matrix has rank <= T-1, so N - (T-1) eigenvalues are
exactly zero. The absorption ratio is still well-defined:
    AR = lambda_1 / sum(non-zero eigenvalues) = lambda_1 / T
The RMT bound still applies with q = T/N.

Functions
---------
    compute_ar(returns_df, window, k)     -> DataFrame (AR, lambda1, n_assets)
    ar_report(returns_df, market)         -> prints summary + correlation with VIX/CSAD
    plot_ar(ar_df, csad_df, vix_s)        -> time series comparison

Usage
-----
    from absorption_ratio import compute_ar, plot_ar
    import numpy as np

    prices  = load_or_fetch_market("sp500", start="2015-01-01")
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    ar      = compute_ar(returns, window=63)
    plot_ar(ar, csad_df=csad_sp, market="sp500")

Dependencies: numpy, pandas, matplotlib
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)

WINDOW   = 21    # rolling window in days (~1 quarter)
K        = 1     # number of eigenvectors for AR (1 = first component only)
MIN_ASSETS = 20  # minimum valid assets per window


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_ar(
    returns_df: pd.DataFrame,
    window: int = WINDOW,
    k: int = K,
    min_assets: int = MIN_ASSETS,
    data_dir: str | None = None,
    market: str = "",
    force: bool = False,
) -> pd.DataFrame:
    """
    Computes rolling Absorption Ratio from a wide returns DataFrame.

    Parameters
    ----------
    returns_df : DataFrame (Date x Ticker) of log returns
    window     : rolling window in trading days (default: 63 ~ 1 quarter)
    k          : number of eigenvectors in numerator (default: 1)
    min_assets : minimum assets with full data per window
    data_dir   : if provided, saves/loads cache CSV
    market     : label for cache filename

    Returns
    -------
    DataFrame with columns:
        AR          - absorption ratio (fraction of variance in top k components)
        lambda1     - first (largest) eigenvalue
        lambda1_rmt - RMT Marchenko-Pastur upper bound for comparison
        excess_lambda1 - lambda1 / lambda1_rmt (>1 = genuine co-movement)
        n_assets    - number of assets used in each window
    """
    if data_dir and market:
        cache = Path(data_dir) / f"ar_{market}_w{window}_k{k}.csv"
        if not force and cache.exists():
            print(f"  [cache] {cache.name}")
            return pd.read_csv(cache, index_col=0, parse_dates=True)

    n_total = returns_df.shape[1]
    rows, dates = [], []

    print(f"  Computing rolling AR  window={window}d  k={k}  assets={n_total}...")

    for end in range(window, len(returns_df) + 1):
        sub = returns_df.iloc[end - window:end]

        # Keep only assets with complete data in this window
        valid = sub.dropna(axis=1, how="any")
        n = valid.shape[1]
        if n < min_assets:
            continue

        # Drop assets with near-zero variance (constant in this window)
        std = valid.std()
        valid = valid.loc[:, std > 1e-8]
        n = valid.shape[1]
        if n < min_assets:
            continue

        # Correlation matrix via pandas (handles edge cases better than np.corrcoef)
        corr_df = valid.corr()
        # Drop any remaining NaN rows/cols (e.g. assets still causing issues)
        corr_df = corr_df.dropna(axis=0, how="any").dropna(axis=1, how="any")
        n = corr_df.shape[0]
        if n < min_assets:
            continue

        corr = corr_df.values
        # Clip to [-1, 1] to avoid numerical issues
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        eigvals = np.linalg.eigvalsh(corr)   # ascending order, real for symmetric
        eigvals = eigvals[::-1]              # descending
        eigvals = eigvals[eigvals > 1e-10]   # remove numerical zeros

        total_var = eigvals.sum()            # = n always (trace identity; zeros contribute 0)
        ar_k      = eigvals[:k].sum() / total_var
        lam1      = eigvals[0]

        # RMT Marchenko-Pastur upper bound
        lam_rmt   = (1 + np.sqrt(n / window)) ** 2

        rows.append({
            "AR":              ar_k,
            "lambda1":         lam1,
            "lambda1_rmt":     lam_rmt,
            "excess_lambda1":  lam1 / lam_rmt,
            "n_assets":        n,
        })
        dates.append(returns_df.index[end - 1])

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    df.index.name = "Date"

    if data_dir and market:
        df.to_csv(cache)
        print(f"  [salvo] {cache.name}  ({len(df)} obs)")

    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def ar_report(
    ar_df: pd.DataFrame,
    csad_df: pd.DataFrame | None = None,
    vix: pd.Series | None = None,
    market: str = "",
) -> None:
    """
    Prints summary statistics and correlations with CSAD and VIX.
    """
    sep   = "=" * 60
    label = market.upper() or "MARKET"

    print(f"\n{sep}")
    print(f"  Absorption Ratio — {label}")
    print(f"  Window: derived from data | k=1 (first eigenvector)")
    print(sep)

    ar = ar_df["AR"]
    l1 = ar_df["lambda1"]
    ex = ar_df["excess_lambda1"]

    print(f"\n  Period     : {ar.index[0].date()} -> {ar.index[-1].date()}")
    print(f"  Obs        : {len(ar)}")
    print(f"\n  AR         : mean={ar.mean():.4f}  std={ar.std():.4f}  "
          f"min={ar.min():.4f}  max={ar.max():.4f}")
    print(f"  lambda1    : mean={l1.mean():.2f}   max={l1.max():.2f}  "
          f"(RMT bound mean={ar_df['lambda1_rmt'].mean():.2f})")
    print(f"  excess_lam1: mean={ex.mean():.2f}   max={ex.max():.2f}  "
          f"(>1 = beyond noise)")

    # Top 5 herding episodes (highest AR)
    print(f"\n  Top 5 herding episodes (highest AR):")
    top5 = ar.nlargest(5)
    for dt, val in top5.items():
        print(f"    {dt.date()}  AR={val:.4f}  lambda1={l1[dt]:.2f}  "
              f"excess={ex[dt]:.2f}x RMT")

    # Correlations with external series
    if csad_df is not None:
        common = ar.index.intersection(csad_df.index)
        corr_csad = ar.loc[common].corr(csad_df.loc[common, "CSAD"])
        print(f"\n  corr(AR, CSAD)   : {corr_csad:+.4f}")
        print(f"  Interpretation   : {'same direction (both rise in stress)' if corr_csad > 0 else 'opposite -- AR and CSAD diverge'}")

    if vix is not None:
        common = ar.index.intersection(vix.index)
        corr_vix = ar.loc[common].corr(vix.loc[common])
        print(f"  corr(AR, VIX)    : {corr_vix:+.4f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_ar(
    ar_df: pd.DataFrame,
    csad_df: pd.DataFrame | None = None,
    vix: pd.Series | None = None,
    market: str = "",
) -> None:
    """
    Three-panel plot:
      1. Absorption Ratio over time
      2. First eigenvalue vs RMT bound
      3. AR vs CSAD (if provided) or AR vs VIX
    """
    n_panels = 2 + (1 if (csad_df is not None or vix is not None) else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=False)
    if n_panels == 2:
        axes = list(axes)

    label = market.upper() or "MARKET"
    fig.suptitle(f"Absorption Ratio — {label}", fontsize=13)

    # Events to annotate
    events = {
        "COVID\n(mar/20)": "2020-03-23",
        "Fed hike\n(mar/22)": "2022-03-16",
        "SVB\n(mar/23)": "2023-03-13",
    }

    def _annotate(ax):
        for txt, dt in events.items():
            d = pd.Timestamp(dt)
            if ar_df.index[0] <= d <= ar_df.index[-1]:
                ax.axvline(d, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
                ylim = ax.get_ylim()
                ax.text(d, ylim[1] * 0.97, txt, fontsize=7,
                        ha="center", va="top", color="gray")

    # Panel 1: Absorption Ratio
    ax = axes[0]
    ax.fill_between(ar_df.index, ar_df["AR"], alpha=0.3, color="#d62728")
    ax.plot(ar_df.index, ar_df["AR"], color="#d62728", linewidth=1.2, label="AR (k=1)")
    ax.plot(ar_df.index, ar_df["AR"].rolling(22).mean(),
            color="darkred", linewidth=1.5, linestyle="--", label="22d MA")
    ax.set_ylabel("Absorption Ratio")
    ax.set_title("Absorption Ratio  (-> 1 = herding, all variance in one component)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _annotate(ax)

    # Panel 2: lambda1 vs RMT bound
    ax = axes[1]
    ax.plot(ar_df.index, ar_df["lambda1"],
            color="#1f77b4", linewidth=1.2, label="lambda_1 (observed)")
    ax.fill_between(ar_df.index, ar_df["lambda1_rmt"],
                    alpha=0.2, color="orange", label="RMT upper bound")
    ax.plot(ar_df.index, ar_df["lambda1_rmt"],
            color="orange", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("First eigenvalue vs RMT bound  (above orange = genuine co-movement)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _annotate(ax)

    # Panel 3: comparison
    if n_panels == 3:
        ax = axes[2]
        if csad_df is not None:
            common = ar_df.index.intersection(csad_df.index)
            # Normalize both to z-score for visual comparison
            ar_z   = (ar_df.loc[common, "AR"] - ar_df["AR"].mean()) / ar_df["AR"].std()
            csad_z = (csad_df.loc[common, "CSAD"] - csad_df["CSAD"].mean()) / csad_df["CSAD"].std()
            ax.plot(common, ar_z,   color="#d62728", linewidth=1.2, label="AR (z-score)", alpha=0.8)
            ax.plot(common, csad_z, color="#1f77b4", linewidth=1.2, label="CSAD (z-score)", alpha=0.8)
            ax.set_title("AR vs CSAD  (z-scores) — divergence = CCK limitation visible", fontsize=10)
            ax.legend(fontsize=8)
        elif vix is not None:
            common = ar_df.index.intersection(vix.index)
            ax.plot(common, ar_df.loc[common, "AR"], color="#d62728",
                    linewidth=1.2, label="AR")
            ax2 = ax.twinx()
            ax2.plot(common, vix.loc[common], color="gray",
                     linewidth=0.8, alpha=0.6, label="VIX")
            ax2.set_ylabel("VIX")
            ax.set_title("AR vs VIX", fontsize=10)
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _annotate(ax)

    fig.tight_layout()
    fname = f"ar_{market}.png" if market else "ar.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from fetch_market import load_or_fetch_market
    from csad import compute_csad

    # Resolve paths relative to this script, not the working directory
    HERE     = Path(__file__).parent
    DATA_DIR = str(HERE / "data" / "raw")
    START    = "2015-01-01"

    # Change working directory so PNGs also save next to the script
    import os; os.chdir(HERE)

    for market in ["sp500", "ibovespa"]:
        print(f"\n{'#'*60}\n  {market.upper()}\n{'#'*60}")

        prices  = load_or_fetch_market(market, start=START, data_dir=DATA_DIR)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        csad    = compute_csad(returns)

        # Load VIX if available
        vix = None
        try:
            ctrl = pd.read_csv(f"{DATA_DIR}/controls_yf_{START}_"
                               f"{datetime.today().strftime('%Y-%m-%d')}.csv",
                               index_col=0, parse_dates=True)
            vix = ctrl["VIX"] if "VIX" in ctrl.columns else None
        except Exception:
            pass

        ar = compute_ar(returns, window=63, k=1,
                        data_dir=DATA_DIR, market=market)

        ar_report(ar, csad_df=csad, vix=vix, market=market)
        plot_ar(ar, csad_df=csad, vix=vix, market=market)
