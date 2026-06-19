"""
csad.py
-------
Calcula o Cross-Sectional Absolute Deviation (CSAD) diario por mercado.

    CSAD_t = (1/N_t) * sum_i |R_i,t - Rm_t|

onde Rm_t = retorno medio equal-weighted cross-sectional.

O CSAD e a metrica central dos Ensaios 2 e 3.
Para o modelo CCK (cck_symmetric.py) serao usadas as colunas:
    CSAD, Rm, abs_Rm, Rm_sq, D_up, D_down

Funcoes:
    compute_csad(returns_df)        -> DataFrame com CSAD + variaveis derivadas
    compute_csad_all(...)           -> dict {mercado: csad_df} com cache

Cache: salva CSV em data_dir/csad_{mercado}_{start}_{end}.csv

Uso em Jupyter:
    from csad import compute_csad, compute_csad_all
    import numpy as np

    prices = load_or_fetch_market("ibovespa", start="2015-01-01")
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    csad = compute_csad(returns)
    csad.tail()

    # Todos os mercados de uma vez
    all_csad = compute_csad_all(start="2015-01-01")
    all_csad["sp500"].describe()
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Mercados com coleta confirmada. Adicionar os demais apos validar os scrapers.
MARKETS = ["sp500", "ibovespa", "nifty50"]


# ---------------------------------------------------------------------------
# Funcao principal
# ---------------------------------------------------------------------------

def compute_csad(
    returns_df: pd.DataFrame,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    Calcula o CSAD diario e variaveis auxiliares para o modelo CCK.

    Parametros
    ----------
    returns_df : DataFrame (Date x Ticker) com retornos log diarios.
                 Valores NaN sao ignorados no calculo (painel desbalanceado ok).
    min_stocks : descarta dias com menos de min_stocks ativos com dado valido.

    Retorna
    -------
    DataFrame com colunas:
        CSAD    - cross-sectional absolute deviation
        Rm      - retorno medio equal-weighted (proxy do mercado)
        abs_Rm  - |Rm|  (regressor linear do CCK)
        Rm_sq   - Rm^2  (regressor nao-linear do CCK; beta2 < 0 -> herding)
        D_up    - indicador: Rm > 0  (usado no CCK assimetrico, Tan et al. 2008)
        D_down  - indicador: Rm < 0
        N       - numero de ativos usados no calculo naquele dia
    """
    # Numero de ativos validos por dia
    n = returns_df.notna().sum(axis=1)
    valid = n >= min_stocks

    Rm   = returns_df.mean(axis=1)                             # skipna=True por padrao
    CSAD = returns_df.sub(Rm, axis=0).abs().mean(axis=1)

    df = pd.DataFrame({
        "CSAD":   CSAD,
        "Rm":     Rm,
        "abs_Rm": Rm.abs(),
        "Rm_sq":  Rm ** 2,
        "D_up":   (Rm > 0).astype(int),
        "D_down": (Rm < 0).astype(int),
        "N":      n,
    }, index=returns_df.index)

    dropped = (~valid).sum()
    if dropped:
        print(f"  [csad] {dropped} dias descartados (N < {min_stocks})")

    return df[valid]


# ---------------------------------------------------------------------------
# Wrapper multi-mercado com cache
# ---------------------------------------------------------------------------

def compute_csad_all(
    markets: list[str] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    min_stocks: int = 10,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Calcula o CSAD para uma lista de mercados, com cache em CSV.

    Parametros
    ----------
    markets   : lista de mercados (default: MARKETS); ver fetch_market.py
    start/end : janela historica
    min_stocks: minimo de ativos por dia para incluir a observacao
    data_dir  : pasta de cache (precos e CSAD)
    force     : re-baixa e re-calcula mesmo se cache existir

    Retorna
    -------
    dict {mercado: DataFrame com CSAD, Rm, abs_Rm, Rm_sq, D_up, D_down, N}
    """
    from fetch_market import load_or_fetch_market

    end     = end or datetime.today().strftime("%Y-%m-%d")
    markets = markets or MARKETS
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    for market in markets:
        cache = Path(data_dir) / f"csad_{market}_{start}_{end}.csv"

        if not force and cache.exists():
            print(f"[{market}] carregando cache: {cache.name}")
            results[market] = pd.read_csv(cache, index_col=0, parse_dates=True)
            continue

        print(f"\n[{market}] calculando CSAD...")
        try:
            prices  = load_or_fetch_market(market, start=start, end=end,
                                           data_dir=data_dir, force=force)
            returns = np.log(prices / prices.shift(1)).dropna(how="all")
            csad    = compute_csad(returns, min_stocks=min_stocks)

            csad.to_csv(cache)
            print(f"  salvo: {cache.name}  ({len(csad)} obs, {int(csad['N'].median())} ativos/dia)")
            results[market] = csad

        except Exception as e:
            print(f"  [ERRO] {market}: {e}")

    return results


# ---------------------------------------------------------------------------
# Estatisticas descritivas relevantes para o CCK
# ---------------------------------------------------------------------------

def csad_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tabela comparativa de estatisticas do CSAD por mercado.

    Retorna
    -------
    DataFrame com linhas = mercados e colunas:
        obs, mean_CSAD, std_CSAD, mean_abs_Rm, std_abs_Rm,
        pct_up, median_N
    """
    rows = []
    for market, df in results.items():
        rows.append({
            "market":      market,
            "obs":         len(df),
            "mean_CSAD":   df["CSAD"].mean(),
            "std_CSAD":    df["CSAD"].std(),
            "mean_abs_Rm": df["abs_Rm"].mean(),
            "std_abs_Rm":  df["abs_Rm"].std(),
            "pct_up":      df["D_up"].mean() * 100,
            "median_N":    df["N"].median(),
        })
    return (
        pd.DataFrame(rows)
        .set_index("market")
        .round(5)
    )


# ---------------------------------------------------------------------------
# Diagnostico de qualidade
# ---------------------------------------------------------------------------

def diagnose(
    csad_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market: str = "",
) -> None:
    """
    Imprime um relatorio de qualidade dos dados de um mercado.

    Verifica:
    - Cobertura temporal e gaps
    - Completude do painel de acoes (% NaN por dia)
    - Outliers nos retornos (> 3 desvios)
    - Distribuicao do CSAD
    - Correlacao CSAD vs |Rm| (deve ser positiva e alta)
    - Dias sem dados (N < threshold)
    """
    returns = np.log(prices_df / prices_df.shift(1)).iloc[1:]

    label = f"  [{market}]" if market else " "
    sep   = "=" * 60

    print(f"\n{sep}")
    print(f"  DIAGNOSTICO: {market.upper() or 'mercado'}")
    print(sep)

    # 1. Cobertura temporal
    print(f"\n  Periodo     : {csad_df.index[0].date()} -> {csad_df.index[-1].date()}")
    print(f"  Obs (dias)  : {len(csad_df)}")

    # 2. Gaps (dias uteis consecutivos com mais de 5 dias de diferenca)
    diffs = csad_df.index.to_series().diff().dt.days.dropna()
    gaps  = diffs[diffs > 5]
    if len(gaps):
        print(f"  Gaps (>5d)  : {len(gaps)} ocorrencias")
        for dt, d in gaps.items():
            print(f"    {dt.date()}  ({int(d)} dias)")
    else:
        print(f"  Gaps        : nenhum")

    # 3. Completude do painel
    pct_nan = returns.isna().mean(axis=1).mean() * 100
    median_n = csad_df["N"].median()
    print(f"\n  Ativos/dia  : mediana={int(median_n)}  total={prices_df.shape[1]}")
    print(f"  NaN medio   : {pct_nan:.1f}% dos valores por dia")

    # 4. Outliers nos retornos
    flat = returns.values.flatten()
    flat = flat[~np.isnan(flat)]
    mu, sigma = flat.mean(), flat.std()
    outliers  = np.sum(np.abs(flat - mu) > 3 * sigma)
    print(f"\n  Retornos    : media={mu:.5f}  std={sigma:.5f}")
    print(f"  Outliers    : {outliers} valores > 3*sigma  ({outliers/len(flat)*100:.2f}%)")

    # 5. CSAD
    c = csad_df["CSAD"]
    print(f"\n  CSAD        : media={c.mean():.5f}  std={c.std():.5f}")
    print(f"  CSAD        : min={c.min():.5f}  max={c.max():.5f}")

    # 6. Correlacao CSAD vs |Rm| — esperado: alta e positiva
    corr_lin = csad_df["CSAD"].corr(csad_df["abs_Rm"])
    corr_sq  = csad_df["CSAD"].corr(csad_df["Rm_sq"])
    print(f"\n  corr(CSAD, |Rm|)  : {corr_lin:.4f}  (esperado: alto e positivo)")
    print(f"  corr(CSAD, Rm^2)  : {corr_sq:.4f}")

    # 7. Simetria up vs down
    pct_up = csad_df["D_up"].mean() * 100
    print(f"\n  Dias alta   : {pct_up:.1f}%  |  Dias baixa: {100-pct_up:.1f}%")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Execucao direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    from fetch_market import load_or_fetch_market

    START    = "2015-01-01"
    DATA_DIR = str(Path(__file__).parent / "data" / "raw")
    MARKET   = "sp500"
    os.chdir(Path(__file__).parent)

    # Carrega precos e calcula CSAD
    prices  = load_or_fetch_market(MARKET, start=START, data_dir=DATA_DIR)
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    csad    = compute_csad(returns)

    # Diagnostico de qualidade
    diagnose(csad, prices, market=MARKET)

    # Resumo multi-mercado (apenas mercados com cache)
    results = compute_csad_all(markets=[MARKET], start=START, data_dir=DATA_DIR)
    print(csad_summary(results).to_string())
