# %% [markdown]
# # Coleta de Dados — Herding, Reflexividade e Preços
#
# Exemplo de uso conjunto de `fetch_market.py`, `fetch_controls.py` e `fetch_institutional.py`.
# Todos os dados são salvos em `./data/raw/` e recarregados do disco nas execuções seguintes.
#
# | Módulo                     | O que faz                                       |
# |----------------------------|-------------------------------------------------|
# | `fetch_market.py`          | Preços dos componentes dos 8 índices            |
# | `fetch_controls.py`        | VIX, DXY, GPR, EPU, séries FRED                |
# | `fetch_institutional.py`   | WGI, Cap/PIB, fluxo estrangeiro (World Bank)    |

# %%
# !pip install yfinance pandas numpy requests beautifulsoup4 openpyxl wbgapi fredapi

import numpy as np
import pandas as pd
from IPython.display import display

from fetch_market import load_or_fetch_market, fetch_asset
from fetch_controls import fetch_yfinance, fetch_fred, fetch_gpr, fetch_epu
from fetch_institutional import fetch_wgi, fetch_market_moderators, fetch_all_institutional

DATA_DIR = "./data/raw"
START    = "2015-01-01"
END      = "2024-12-31"

# %%
# ---
# ## 1. Dados de Mercado
#
# `load_or_fetch_market` baixa os componentes de um índice e salva em CSV.
# Na segunda execução, carrega do disco sem fazer nenhum request.
# Componentes do Ibovespa via API da B3
ibov = load_or_fetch_market("ibovespa", start=START, end=END, data_dir=DATA_DIR)
print(ibov.shape)
ibov.tail(3)

# %%
# Componentes do S&P 500 via Wikipedia
sp500 = load_or_fetch_market("sp500", start=START, end=END, data_dir=DATA_DIR)
print(sp500.shape)

# %%
# Ativo individual — índice ^GSPC para comparação
gspc = fetch_asset("^GSPC", start=START, end=END, verbose=False)
gspc[["Adj_Close", "Log_Return", "Cumulative_Return"]].tail(3)

# %% [markdown]
# ### Retornos logarítmicos e preview do CSAD
#
# O CSAD (Cross-Sectional Absolute Deviation) é a métrica central dos Ensaios 2 e 3:
#
# $$\text{CSAD}_t = \frac{1}{N} \sum_{i=1}^{N} |R_{i,t} - R_{m,t}|$$

# %%
# how="all" descarta só linhas onde TODOS os tickers são NaN (ex: feriados globais)
# how="any" (padrão) descartaria qualquer linha com um ticker faltando — esvazia o df
ret_ibov  = np.log(ibov  / ibov.shift(1)).dropna(how="all")
ret_sp500 = np.log(sp500 / sp500.shift(1)).dropna(how="all")

Rm_ibov   = ret_ibov.mean(axis=1)
CSAD_ibov = ret_ibov.sub(Rm_ibov, axis=0).abs().mean(axis=1)

pd.DataFrame({"Rm": Rm_ibov, "CSAD": CSAD_ibov}).describe().round(5)

# %%
# Estatísticas descritivas dos retornos por ativo
ret_ibov.describe().T[["mean", "std", "min", "max"]].round(5).head(10)

# %% [markdown]
# ---
# ## 2. Controles Macrofinanceiros
#
# | Função           | Requer chave? | O que retorna                          |
# |------------------|:---:|----------------------------------------|
# | `fetch_yfinance` | Não | VIX, DXY                               |
# | `fetch_gpr`      | Não | GPR (Caldara & Iacoviello)             |
# | `fetch_epu`      | Não | EPU (Baker, Bloom & Davis)             |
# | `fetch_fred`     | Sim (gratuita) | MOVE, DFF, SOFR, SONIA, T10Y2Y, EMBI |

# %%
# VIX e DXY via yfinance
yf_ctrl = fetch_yfinance(start=START, end=END, data_dir=DATA_DIR)
yf_ctrl.tail(3)

# %%
# GPR mensal (Caldara & Iacoviello, 2022)
gpr = fetch_gpr(freq="monthly", start=START, end=END, data_dir=DATA_DIR)
gpr[["GPR", "GPR_THREAT", "GPR_ACT"]].tail(5)

# %%
# EPU mensal — EUA (Baker, Bloom & Davis, 2016)
epu = fetch_epu(country="US", start=START, end=END, data_dir=DATA_DIR)
epu.tail(3)

# %%
# FRED — chave gratuita: https://fred.stlouisfed.org/docs/api/api_key.html
# Defina aqui ou via variável de ambiente: FRED_API_KEY=sua_chave

FRED_KEY = "b039a8bb99b9b17f710bc7447513eb3b"

if FRED_KEY:
    fred_ctrl = fetch_fred(start=START, end=END, api_key=FRED_KEY, data_dir=DATA_DIR)
    display(fred_ctrl.tail(3))
else:
    print("FRED_KEY não definida. Séries disponíveis quando configurado:")
    print("  MOVE (bond vol), DFF (Fed Funds), SOFR, SONIA, T10Y2Y (curva), EMBI")

# %% [markdown]
# ---
# ## 3. Moderadores Institucionais (Ensaio 2)
#
# Dados **anuais** do Banco Mundial para os 8 mercados da tese.
# Output com `MultiIndex (country, year)` — pronto para `linearmodels.panel.PanelOLS`.

# %%
# WGI — 6 dimensões de qualidade institucional
wgi = fetch_wgi(start_year=2000, data_dir=DATA_DIR)
wgi.head(10)

# %%
# Consolidado: WGI + Cap/PIB + fluxo estrangeiro
inst = fetch_all_institutional(start_year=2000, data_dir=DATA_DIR)

# Corte transversal em 2022
inst.xs(2022, level="year")[["Regulatory_Quality", "MktCap_GDP_pct"]]

# %%
# Série temporal do Brasil
inst.loc["BRA", ["Regulatory_Quality", "MktCap_GDP_pct"]].tail(10)

# %% [markdown]
# ---
# ## 4. Alinhamento para o Painel
#
# Os dados têm frequências diferentes:
# - Retornos / CSAD → **diário**
# - GPR, EPU        → **mensal**
# - WGI, Cap/PIB    → **anual**
#
# Estratégia: agregar CSAD para mensal/anual, ou expandir moderadores via `ffill`.

# %%
# Dataset diário: CSAD + Rm + VIX + DXY
daily = pd.concat([
    Rm_ibov.rename("Rm"),
    CSAD_ibov.rename("CSAD"),
    yf_ctrl[["VIX", "DXY"]],
], axis=1).dropna()

print(daily.shape)
daily.tail(5)

# %%
# Dataset mensal: CSAD + GPR + moderadores institucionais (Brasil)
csad_monthly = CSAD_ibov.resample("ME").mean()

# Moderadores anuais → mensais via forward-fill
bra_inst = inst.loc["BRA", ["Regulatory_Quality", "MktCap_GDP_pct"]].copy()
bra_inst.index = pd.to_datetime(bra_inst.index.astype(str))
bra_inst_monthly = bra_inst.resample("ME").last().ffill()

gpr_monthly = gpr["GPR"].resample("ME").last()

panel_bra = pd.concat([
    csad_monthly.rename("CSAD"),
    gpr_monthly.rename("GPR"),
    bra_inst_monthly,
], axis=1).dropna()

print(panel_bra.shape)
panel_bra.tail(6)

# %% [markdown]
# ---
# ## Próximos passos
#
# ```
# src/metrics/
#   csad.py               → compute_csad(returns_df) para todos os mercados
#   realized_vol.py       → volatilidade realizada (janela 22d)
#   garch.py              → GARCH(1,1) por mercado
#
# src/essay2/
#   cck_symmetric.py      → OLS: CSAD ~ |Rm| + Rm²  (Newey-West, lags=5)
#   cck_asymmetric.py     → separa dias up/down (Tan et al. 2008)
#   panel_institutional.py → PanelOLS com efeitos fixos bidirecionais
# ```
