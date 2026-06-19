# %% [markdown]
# # CSAD e CCK — S&P 500 vs Ibovespa
#
# Visualiza e compara o Cross-Sectional Absolute Deviation (CSAD)
# dos dois mercados e mostra os resultados do modelo CCK.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display

from fetch_market import load_or_fetch_market
from csad import compute_csad
from cck_symmetric import cck_report

DATA_DIR = "./data/raw"
START    = "2015-01-01"
END      = "2024-12-31"

# %% [markdown]
# ## 1. Carrega precos e calcula CSAD

# %%
prices_sp  = load_or_fetch_market("sp500",    start=START, end=END, data_dir=DATA_DIR)
prices_ibov = load_or_fetch_market("ibovespa", start=START, end=END, data_dir=DATA_DIR)

ret_sp   = np.log(prices_sp   / prices_sp.shift(1)).dropna(how="all")
ret_ibov = np.log(prices_ibov / prices_ibov.shift(1)).dropna(how="all")

csad_sp   = compute_csad(ret_sp)
csad_ibov = compute_csad(ret_ibov)

print("S&P 500 :", csad_sp.shape,   "| ativos/dia:", int(csad_sp["N"].median()))
print("Ibovespa:", csad_ibov.shape, "| ativos/dia:", int(csad_ibov["N"].median()))

# %% [markdown]
# ## 2. CSAD — serie temporal comparada

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# Janela rolante de 22 dias para suavizar
roll = 22

for ax, csad, label, color in [
    (axes[0], csad_sp,   "S&P 500",  "#1f77b4"),
    (axes[1], csad_ibov, "Ibovespa", "#d62728"),
]:
    ax.fill_between(csad.index, csad["CSAD"] * 100, alpha=0.25, color=color)
    ax.plot(csad.index,
            csad["CSAD"].rolling(roll).mean() * 100,
            color=color, linewidth=1.4, label=f"CSAD {roll}d MA")

    # Anota eventos relevantes
    events = {
        "COVID\n(mar/20)": "2020-03-23",
        "Fed hike\n(mar/22)": "2022-03-16",
        "SVB\n(mar/23)": "2023-03-13",
    }
    for txt, dt in events.items():
        d = pd.Timestamp(dt)
        if csad.index[0] <= d <= csad.index[-1]:
            ax.axvline(d, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(d, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 3,
                    txt, fontsize=7, ha="center", va="top", color="gray")

    ax.set_title(f"CSAD diario — {label}  (linha = media {roll}d)", fontsize=12)
    ax.set_ylabel("CSAD (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.tight_layout()
plt.savefig("csad_sp500_ibov_timeseries.png", dpi=150)
plt.show()
print("Salvo: csad_sp500_ibov_timeseries.png")

# %% [markdown]
# ## 3. CSAD normalizado — comparacao direta

# %%
# Alinha os dois em datas comuns e normaliza pelo desvio padrao
common = csad_sp.index.intersection(csad_ibov.index)

sp_norm   = (csad_sp.loc[common,   "CSAD"] - csad_sp["CSAD"].mean())   / csad_sp["CSAD"].std()
ibov_norm = (csad_ibov.loc[common, "CSAD"] - csad_ibov["CSAD"].mean()) / csad_ibov["CSAD"].std()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(common, sp_norm.rolling(22).mean(),   label="S&P 500",  linewidth=1.3, color="#1f77b4")
ax.plot(common, ibov_norm.rolling(22).mean(), label="Ibovespa", linewidth=1.3, color="#d62728", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
ax.fill_between(common, sp_norm.rolling(22).mean(), ibov_norm.rolling(22).mean(),
                alpha=0.1, color="gray", label="diferenca")
ax.set_title("CSAD normalizado (z-score) — S&P 500 vs Ibovespa  (media 22d)", fontsize=12)
ax.set_ylabel("CSAD (desvios da media)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("csad_normalizado_comparacao.png", dpi=150)
plt.show()
print("Salvo: csad_normalizado_comparacao.png")

# %% [markdown]
# ## 4. Dispersao em dias extremos — scatter CSAD vs |Rm|

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, csad, label, color in [
    (axes[0], csad_sp,   "S&P 500",  "#1f77b4"),
    (axes[1], csad_ibov, "Ibovespa", "#d62728"),
]:
    x = csad["abs_Rm"] * 100
    y = csad["CSAD"]   * 100

    # Colore por direcao do mercado
    up   = csad["D_up"] == 1
    ax.scatter(x[up],  y[up],  alpha=0.15, s=4, color="green",  label="alta")
    ax.scatter(x[~up], y[~up], alpha=0.15, s=4, color="red",    label="baixa")

    # Linha de tendencia (quadratica)
    z = np.polyfit(x, y, 2)
    xr = np.linspace(x.min(), x.quantile(0.99), 200)
    ax.plot(xr, np.polyval(z, xr), color=color, linewidth=2, label="fit quadratico")

    ax.set_title(f"CSAD vs |Rm| — {label}", fontsize=11)
    ax.set_xlabel("|Rm| (%)")
    ax.set_ylabel("CSAD (%)")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

fig.tight_layout()
plt.savefig("csad_scatter_sp500_ibov.png", dpi=150)
plt.show()
print("Salvo: csad_scatter_sp500_ibov.png")

# %% [markdown]
# ## 5. Resultados CCK

# %%
print("\n" + "="*65)
print("  RESULTADOS CCK")
print("="*65)
res_sp   = cck_report(csad_sp,   market="sp500")
res_ibov = cck_report(csad_ibov, market="ibovespa")

# %% [markdown]
# ## 6. Tabela comparativa de coeficientes

# %%
rows = []
for label, res in [("S&P 500", res_sp), ("Ibovespa", res_ibov)]:
    sym = res["symmetric"]
    rows.append({
        "Mercado":  label,
        "alpha":    round(sym.params["alpha"],  5),
        "beta1":    round(sym.params["beta1"],  5),
        "beta2":    round(sym.params["beta2"],  5),
        "p(beta2)": round(sym.pvalues["beta2"], 4),
        "R2":       round(sym.rsquared,          4),
        "Herding":  "sim (beta2<0, p<0.05)" if sym.params["beta2"] < 0 and sym.pvalues["beta2"] < 0.05 else "nao",
    })

tbl = pd.DataFrame(rows).set_index("Mercado")
display(tbl)
