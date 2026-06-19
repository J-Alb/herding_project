"""
cck_symmetric.py
----------------
Estima as especificacoes CCK simetrica e assimetrica do modelo de herding.

Especificacao simetrica (Chang, Cheng & Khorana 2000):
    CSAD_t = alpha + beta1*|Rm_t| + beta2*Rm_t^2 + e_t

    Hipotese: beta2 < 0 e significativo -> herding detectado.
    Intuicao: sob herding, a dispersao cross-sectional cresce menos do que
    linearmente com o retorno de mercado (ou ate cai), gerando um termo
    quadratico negativo.

Especificacao assimetrica (Tan, Chiang, Mason & Nelling 2008):
    CSAD_t = alpha + beta1_up*D_up*|Rm| + beta1_dn*D_dn*|Rm|
                   + beta2_up*D_up*Rm^2  + beta2_dn*D_dn*Rm^2 + e_t

    Hipotese: beta2_dn < 0 (herding em dias de queda) e |beta2_dn| > |beta2_up|
    (herding mais forte em mercados em baixa — tipico em emergentes).

Erros-padrao: Newey-West (HAC) com lags=5 (1 semana util).

Funcoes:
    fit_symmetric(csad_df)       -> OLSResults com NW se
    fit_asymmetric(csad_df)      -> OLSResults com NW se
    cck_report(csad_df, market)  -> imprime tabela completa

Uso em Jupyter:
    from cck_symmetric import cck_report
    from csad import compute_csad
    import numpy as np

    prices  = load_or_fetch_market("sp500", start="2015-01-01")
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    csad    = compute_csad(returns)
    cck_report(csad, market="sp500")

Dependencias:
    pip install statsmodels
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

NW_LAGS = 5  # Newey-West lags (1 semana util)


# ---------------------------------------------------------------------------
# Estimacao
# ---------------------------------------------------------------------------

def fit_symmetric(
    csad_df: pd.DataFrame,
    nw_lags: int = NW_LAGS,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Estima CSAD_t = alpha + beta1*|Rm_t| + beta2*Rm_t^2 + e_t
    com erros-padrao Newey-West.

    Parametros
    ----------
    csad_df : DataFrame retornado por compute_csad() — precisa de
              colunas CSAD, abs_Rm, Rm_sq
    nw_lags : numero de lags para a correccao Newey-West (default: 5)

    Retorna
    -------
    statsmodels RegressionResults com cov_type='HAC'
    """
    df = csad_df[["CSAD", "abs_Rm", "Rm_sq"]].dropna()

    y = df["CSAD"]
    X = sm.add_constant(df[["abs_Rm", "Rm_sq"]])
    X.columns = ["alpha", "beta1", "beta2"]

    model = sm.OLS(y, X)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})


def fit_asymmetric(
    csad_df: pd.DataFrame,
    nw_lags: int = NW_LAGS,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Estima especificacao assimetrica Tan et al. (2008):
    CSAD_t = alpha
           + beta1_up * D_up  * |Rm|
           + beta1_dn * D_dn  * |Rm|
           + beta2_up * D_up  * Rm^2
           + beta2_dn * D_dn  * Rm^2
           + e_t

    com erros-padrao Newey-West.
    """
    df = csad_df[["CSAD", "abs_Rm", "Rm_sq", "D_up", "D_down"]].dropna()

    df = df.copy()
    df["b1_up"] = df["D_up"]   * df["abs_Rm"]
    df["b1_dn"] = df["D_down"] * df["abs_Rm"]
    df["b2_up"] = df["D_up"]   * df["Rm_sq"]
    df["b2_dn"] = df["D_down"] * df["Rm_sq"]

    y = df["CSAD"]
    X = sm.add_constant(df[["b1_up", "b1_dn", "b2_up", "b2_dn"]])
    X.columns = ["alpha", "beta1_up", "beta1_dn", "beta2_up", "beta2_dn"]

    model = sm.OLS(y, X)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _fmt_row(name: str, coef: float, se: float, t: float, p: float) -> str:
    return (f"  {name:<12s}  {coef:>10.5f}  {se:>10.5f}  "
            f"{t:>8.3f}  {p:>7.4f}  {_stars(p)}")


def cck_report(
    csad_df: pd.DataFrame,
    market: str = "",
    nw_lags: int = NW_LAGS,
) -> dict:
    """
    Estima e imprime os dois modelos CCK para um mercado.

    Retorna
    -------
    dict com chaves "symmetric" e "asymmetric" (RegressionResults)
    """
    sep   = "=" * 65
    label = market.upper() or "MERCADO"
    n     = len(csad_df.dropna())
    period = f"{csad_df.index[0].date()} -> {csad_df.index[-1].date()}"

    print(f"\n{sep}")
    print(f"  CCK — {label}  |  {period}  |  N={n}")
    print(sep)

    # --- Simetrico ---
    res_sym = fit_symmetric(csad_df, nw_lags=nw_lags)
    b2      = res_sym.params["beta2"]
    p_b2    = res_sym.pvalues["beta2"]
    herding = "HERDING DETECTADO" if (b2 < 0 and p_b2 < 0.05) else "sem herding significativo"

    print(f"\n  [1] Simetrico: CSAD = alpha + beta1*|Rm| + beta2*Rm^2")
    print(f"  Erros-padrao: Newey-West (lags={nw_lags})\n")
    print(f"  {'Param':<12s}  {'Coef':>10s}  {'Std Err':>10s}  {'t':>8s}  {'p-val':>7s}")
    print(f"  {'-'*60}")
    for nm in ["alpha", "beta1", "beta2"]:
        print(_fmt_row(nm, res_sym.params[nm], res_sym.bse[nm],
                       res_sym.tvalues[nm], res_sym.pvalues[nm]))
    print(f"\n  R2={res_sym.rsquared:.4f}   R2-adj={res_sym.rsquared_adj:.4f}"
          f"   F={res_sym.fvalue:.2f}  (p={res_sym.f_pvalue:.4f})")
    print(f"\n  >> {herding}  (beta2={b2:.5f}, p={p_b2:.4f})")

    # --- Assimetrico ---
    res_asy = fit_asymmetric(csad_df, nw_lags=nw_lags)
    b2_up   = res_asy.params["beta2_up"]
    b2_dn   = res_asy.params["beta2_dn"]
    p_up    = res_asy.pvalues["beta2_up"]
    p_dn    = res_asy.pvalues["beta2_dn"]

    print(f"\n{'-'*65}")
    print(f"\n  [2] Assimetrico (Tan et al. 2008):")
    print(f"  CSAD = alpha + beta1_up*D+*|Rm| + beta1_dn*D-*|Rm|")
    print(f"               + beta2_up*D+*Rm^2 + beta2_dn*D-*Rm^2\n")
    print(f"  {'Param':<12s}  {'Coef':>10s}  {'Std Err':>10s}  {'t':>8s}  {'p-val':>7s}")
    print(f"  {'-'*60}")
    for nm in ["alpha", "beta1_up", "beta1_dn", "beta2_up", "beta2_dn"]:
        print(_fmt_row(nm, res_asy.params[nm], res_asy.bse[nm],
                       res_asy.tvalues[nm], res_asy.pvalues[nm]))
    print(f"\n  R2={res_asy.rsquared:.4f}   R2-adj={res_asy.rsquared_adj:.4f}"
          f"   F={res_asy.fvalue:.2f}  (p={res_asy.f_pvalue:.4f})")

    asym_note = []
    if b2_dn < 0 and p_dn < 0.05:
        asym_note.append("herding em baixa (beta2_dn < 0, p<0.05)")
    if b2_up < 0 and p_up < 0.05:
        asym_note.append("herding em alta (beta2_up < 0, p<0.05)")
    if b2_dn < b2_up:
        asym_note.append(f"assimetria: beta2_dn ({b2_dn:.5f}) < beta2_up ({b2_up:.5f})")
    print(f"\n  >> " + ("  |  ".join(asym_note) if asym_note else "sem resultados assimetricos significativos"))

    print(f"\n{sep}\n")
    return {"symmetric": res_sym, "asymmetric": res_asy}


# ---------------------------------------------------------------------------
# Execucao direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    from fetch_market import load_or_fetch_market
    from csad import compute_csad

    START    = "2015-01-01"
    DATA_DIR = str(Path(__file__).parent / "data" / "raw")
    os.chdir(Path(__file__).parent)

    prices  = load_or_fetch_market("sp500", start=START, data_dir=DATA_DIR)
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    csad    = compute_csad(returns)

    results = cck_report(csad, market="sp500")

    # Salva coeficientes em CSV para uso posterior no painel
    sym = results["symmetric"]
    asy = results["asymmetric"]

    rows = []
    for nm in ["alpha", "beta1", "beta2"]:
        rows.append({"model": "symmetric", "param": nm,
                     "coef": sym.params[nm], "se": sym.bse[nm],
                     "t": sym.tvalues[nm], "p": sym.pvalues[nm]})
    for nm in ["alpha", "beta1_up", "beta1_dn", "beta2_up", "beta2_dn"]:
        rows.append({"model": "asymmetric", "param": nm,
                     "coef": asy.params[nm], "se": asy.bse[nm],
                     "t": asy.tvalues[nm], "p": asy.pvalues[nm]})

    out = pd.DataFrame(rows)
    out.to_csv(f"{DATA_DIR}/cck_sp500_{START[:4]}.csv", index=False)
    print(f"Coeficientes salvos em {DATA_DIR}/cck_sp500_{START[:4]}.csv")
