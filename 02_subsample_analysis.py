# %%
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display

from cck_symmetric import fit_symmetric, fit_asymmetric, _stars
from cck_quantile   import fit_quantile_cck

DATA_DIR   = "./data/raw"
MARKET     = "sp500"
SUB_START  = "2019-01-01"
SUB_END    = "2024-12-31"
CCK_END    = "2024-12-31"   # CCK analysis fixed to paper sample
QR_QUANTS  = [0.10, 0.25, 0.50, 0.75, 0.90]

# %%
csad = pd.read_csv(
    sorted(glob.glob(f"{DATA_DIR}/csad_{MARKET}_*.csv"))[-1],
    index_col=0, parse_dates=True
)
ar = pd.read_csv(
    sorted(glob.glob(f"{DATA_DIR}/ar_{MARKET}_*.csv"))[-1],
    index_col=0, parse_dates=True
)

csad_sub = csad.loc[SUB_START:SUB_END]
ar_sub   = ar.loc[SUB_START:SUB_END]

print(f"Full : {csad.index[0].date()} -> {csad.index[-1].date()}  n={len(csad)}")
print(f"Sub  : {csad_sub.index[0].date()} -> {csad_sub.index[-1].date()}  n={len(csad_sub)}")

# %%
common = csad.index.intersection(ar.index)
csad_a = csad.loc[common]
ar_a   = ar.loc[common]

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

ax1.fill_between(csad_a.index, csad_a["CSAD"] * 100,
                 alpha=0.20, color="#1f77b4")
ax1.plot(csad_a.index, csad_a["CSAD"].rolling(22).mean() * 100,
         color="#1f77b4", linewidth=1.4, label="CSAD 22d MA (%)")

ax2.plot(ar_a.index, ar_a["excess_lambda1"],
         color="#d62728", linewidth=1.2, alpha=0.85, label="lambda1 / RMT bound")
mean_excess = ar_a["excess_lambda1"].mean()
ax2.axhline(mean_excess, color="orange", linewidth=1.0, linestyle="--", alpha=0.8,
            label=f"Historical mean ({mean_excess:.1f}x)")

ax1.axvspan(pd.Timestamp(SUB_START), pd.Timestamp(SUB_END),
            alpha=0.06, color="green", label="sub-sample")

for txt, dt in [("COVID\n(mar/20)",        "2020-03-23"),
                ("Fed\n(mar/22)",          "2022-03-16"),
                ("SVB\n(mar/23)",          "2023-03-13"),
                ("Liberation\nDay(apr/25)","2025-04-02")]:
    d = pd.Timestamp(dt)
    if csad_a.index[0] <= d <= csad_a.index[-1]:
        ax1.axvline(d, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax1.text(d, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1 else 3,
                 txt, fontsize=7, ha="center", va="top", color="gray")

ax1.set_ylabel("CSAD (%)", color="#1f77b4")
ax2.set_ylabel("lambda1 / RMT bound  (> 1 = genuine co-movement)", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.set_title(f"CSAD vs Excess lambda1 (lambda1 / RMT bound) — {MARKET.upper()}  (green band = sub-sample)", fontsize=12)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax1.grid(alpha=0.3)

fig.tight_layout()
plt.savefig(f"csad_lambda1_{MARKET}.png", dpi=150)
plt.show()

# %%
def _cck_table(csad_df, label):
    res = fit_symmetric(csad_df)
    rows = []
    for nm in ["alpha", "beta1", "beta2"]:
        rows.append({
            "param":   nm,
            "coef":    round(res.params[nm], 6),
            "se":      round(res.bse[nm], 6),
            "t":       round(res.tvalues[nm], 3),
            "p":       round(res.pvalues[nm], 4),
            "sig":     _stars(res.pvalues[nm]),
        })
    df = pd.DataFrame(rows).set_index("param")
    df.columns = pd.MultiIndex.from_tuples([(label, c) for c in df.columns])
    return df, res

csad_cck  = csad.loc[:CCK_END]
tbl_full, res_full = _cck_table(csad_cck,  f"Full ({len(csad_cck)} obs)")
tbl_sub,  res_sub  = _cck_table(csad_sub,  f"Sub  ({len(csad_sub)} obs)")

comparison = pd.concat([tbl_full, tbl_sub], axis=1)
display(comparison)

b2_full = res_full.params["beta2"];  p2_full = res_full.pvalues["beta2"]
b2_sub  = res_sub.params["beta2"];   p2_sub  = res_sub.pvalues["beta2"]
print(f"\nbeta2 full: {b2_full:+.5f}  ({_stars(p2_full) or 'ns'})")
print(f"beta2 sub : {b2_sub:+.5f}  ({_stars(p2_sub) or 'ns'})")

# %%
def _qr_beta2(csad_df, quantiles):
    res = fit_quantile_cck(csad_df, quantiles=quantiles)
    b2  = res[res["param"] == "beta2"].set_index("quantile")
    return b2

qr_full = _qr_beta2(csad_cck, QR_QUANTS)
qr_sub  = _qr_beta2(csad_sub, QR_QUANTS)

fig, ax = plt.subplots(figsize=(9, 5))

for df, label, color in [(qr_full, f"Full sample",        "#1f77b4"),
                          (qr_sub,  f"Sub {SUB_START[:4]}-{SUB_END[:4]}", "#d62728")]:
    q  = df.index.values
    c  = df["coef"].values
    lo = df["lower_ci"].values
    hi = df["upper_ci"].values
    ax.plot(q, c, "o-", color=color, linewidth=2, markersize=7, label=label)
    ax.fill_between(q, lo, hi, alpha=0.15, color=color)
    for i, row in df.iterrows():
        s = _stars(row["p_value"])
        if s:
            ax.text(row.name, row["coef"] + (hi[list(q).index(row.name)] - row["coef"]) * 0.3,
                    s, ha="center", fontsize=8, color=color)

ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax.set_xticks(QR_QUANTS)
ax.set_xlabel("Quantile (tau)")
ax.set_ylabel("beta2  [Rm^2 coefficient]")
ax.set_title(f"beta2 across quantiles — full vs sub-sample  ({MARKET.upper()})\n"
             f"Below zero = herding at that quantile", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig(f"cck_quantile_comparison_{MARKET}.png", dpi=150)
plt.show()

# %%
print(f"beta2 path (full): " + "  ".join(f"tau={q:.2f}: {r['coef']:+.4f}" for q, r in qr_full.iterrows()))
print(f"beta2 path (sub) : " + "  ".join(f"tau={q:.2f}: {r['coef']:+.4f}" for q, r in qr_sub.iterrows()))

crossed_full = any(qr_full["coef"] < 0)
crossed_sub  = any(qr_sub["coef"]  < 0)
print(f"\nFull sample — beta2 crosses zero: {crossed_full}")
print(f"Sub-sample  — beta2 crosses zero: {crossed_sub}")
