"""
fetch_institutional.py
----------------------
Coleta moderadores institucionais para o Ensaio 2 da tese.

Dados via wbgapi (World Bank API):
  WGI  : 6 dimensões de qualidade institucional (Rule of Law, Regulatory Quality, ...)
  Cap/PIB : capitalização de mercado / PIB  (CM.MKT.LCAP.GD.ZS)
  Fluxos  : portfólio estrangeiro líquido   (BX.PEF.TOTL.CD.WD)  - proxy para
            participação estrangeira (EPFR Global / BIS requerem acesso manual)

Frequência: anual (World Bank publica anualmente).
Output: DataFrame com MultiIndex (country, year) - formato pronto para
        linearmodels.panel.PanelOLS.

Cache: salva CSV em data_dir/; re-usa se já existir (force=True para re-baixar).

Dependências:
    pip install wbgapi pandas

Uso em Jupyter:
    from fetch_institutional import fetch_all_institutional
    inst = fetch_all_institutional(start_year=2000, end_year=2023)
    inst.loc["BRA"]          # série temporal do Brasil
    inst.xs(2020, level="year")  # corte transversal em 2020
"""

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Países e indicadores da tese
# ---------------------------------------------------------------------------

THESIS_COUNTRIES = ["USA", "GBR", "DEU", "JPN", "BRA", "MEX", "CHN", "IND"]

COUNTRY_NAMES = {
    "USA": "EUA",
    "GBR": "Reino Unido",
    "DEU": "Alemanha",
    "JPN": "Japão",
    "BRA": "Brasil",
    "MEX": "México",
    "CHN": "China",
    "IND": "Índia",
}

# World Governance Indicators (WGI) - estimativas anuais, escala ~ [-2.5, +2.5]
WGI_INDICATORS = {
    "GOV_WGI_RL.EST": "Rule_of_Law",
    "GOV_WGI_RQ.EST": "Regulatory_Quality",
    "GOV_WGI_GE.EST": "Govt_Effectiveness",
    "GOV_WGI_CC.EST": "Control_of_Corruption",
    "GOV_WGI_PV.EST": "Political_Stability",
    "GOV_WGI_VA.EST": "Voice_Accountability",
}

# Outros indicadores World Bank
WB_INDICATORS = {
    "CM.MKT.LCAP.GD.ZS": "MktCap_GDP_pct",       # capitalização / PIB (%)
    "BX.PEF.TOTL.CD.WD": "Portfolio_Inflows_USD",  # fluxo líquido de portfólio (USD)
    "NY.GDP.MKTP.CD":     "GDP_USD",                # PIB corrente (USD) - denominador útil
}


# ---------------------------------------------------------------------------
# Helper: World Bank -> DataFrame longo (country, year, valor)
# ---------------------------------------------------------------------------

def _fetch_wb(
    indicator: str,
    col_name: str,
    countries: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Baixa um indicador do World Bank via API REST v2 (sem wbgapi).
    Retorna DataFrame com colunas [country, year, col_name].
    """
    import requests

    country_str = ";".join(countries)
    base = (
        f"https://api.worldbank.org/v2/country/{country_str}"
        f"/indicator/{indicator}"
        f"?format=json&per_page=1000&date={start_year}:{end_year}"
    )

    rows = []
    page = 1
    while True:
        r = requests.get(f"{base}&page={page}",
                         headers={"User-Agent": "research/1.0"}, timeout=30)
        r.raise_for_status()
        payload = r.json()

        meta = payload[0]
        data = payload[1] if len(payload) > 1 else []
        for rec in (data or []):
            if rec.get("value") is not None and rec.get("countryiso3code"):
                rows.append({
                    "country": rec["countryiso3code"],
                    "year":    int(rec["date"]),
                    col_name:  float(rec["value"]),
                })

        if page >= meta.get("pages", 1):
            break
        page += 1
        time.sleep(0.2)

    return pd.DataFrame(rows).sort_values(["country", "year"])


def _merge_long(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Une vários DataFrames longos em (country, year) e retorna MultiIndex."""
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["country", "year"], how="outer")
    return result.set_index(["country", "year"]).sort_index()


# ---------------------------------------------------------------------------
# Cache helper (mesma lógica dos outros scripts)
# ---------------------------------------------------------------------------

def _cached(path: Path, fn, force: bool) -> pd.DataFrame:
    if not force and path.exists():
        print(f"  [cache] {path.name}")
        df = pd.read_csv(path, index_col=["country", "year"])
        return df
    df = fn()
    df.to_csv(path)
    print(f"  [salvo] {path.name}  ({df.shape[1]} colunas, {df.shape[0]} obs)")
    return df


# ---------------------------------------------------------------------------
# 1. WGI - World Governance Indicators
# ---------------------------------------------------------------------------

def fetch_wgi(
    countries: list[str] | None = None,
    start_year: int = 2000,
    end_year: int | None = None,
    indicators: dict[str, str] | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Baixa os World Governance Indicators (Banco Mundial).

    Parâmetros
    ----------
    countries  : lista de códigos ISO-3 (default: 8 mercados da tese)
    start_year : primeiro ano
    end_year   : último ano (default: ano atual)
    indicators : dict {wb_code: nome_coluna}; None -> WGI_INDICATORS completo

    Retorna
    -------
    DataFrame com MultiIndex (country, year) e uma coluna por indicador WGI.
    Escala: estimativas padronizadas ~ [-2.5, +2.5].

    Exemplo
    -------
    wgi = fetch_wgi()
    wgi.loc["BRA", "Regulatory_Quality"]   # série temporal do Brasil
    wgi.xs(2022, level="year")              # corte transversal em 2022
    """
    from datetime import datetime
    countries  = countries  or THESIS_COUNTRIES
    indicators = indicators or WGI_INDICATORS
    end_year   = end_year   or datetime.today().year - 1  # WB tem lag de ~1 ano

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"wgi_{start_year}_{end_year}.csv"

    def _download():
        frames = []
        for wb_code, col_name in indicators.items():
            try:
                df = _fetch_wb(wb_code, col_name, countries, start_year, end_year)
                frames.append(df)
                print(f"  WGI  {wb_code:<12s} ({col_name}) -> {len(df)} obs")
            except Exception as e:
                print(f"  [erro] WGI {wb_code}: {e}")
        if not frames:
            raise RuntimeError("Nenhum indicador WGI baixado.")
        return _merge_long(frames)

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 2. Moderadores de mercado - Cap/PIB e fluxos estrangeiros
# ---------------------------------------------------------------------------

def fetch_market_moderators(
    countries: list[str] | None = None,
    start_year: int = 2000,
    end_year: int | None = None,
    indicators: dict[str, str] | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Baixa moderadores de desenvolvimento de mercado via World Bank:
      - MktCap_GDP_pct     : capitalização / PIB (%)      [CM.MKT.LCAP.GD.ZS]
      - Portfolio_Inflows  : fluxo líquido de portfólio   [BX.PEF.TOTL.CD.WD]
      - GDP_USD            : PIB corrente em USD           [NY.GDP.MKTP.CD]

    Nota sobre participação estrangeira
    ------------------------------------
    BX.PEF.TOTL.CD.WD é um proxy de fluxo (BoP). Para participação estrangeira
    como % da capitalização (fluxo estrangeiro / cap), divida por GDP_USD ×
    MktCap_GDP_pct. Dados EPFR Global / BIS requerem acesso manual.

    Retorna
    -------
    DataFrame com MultiIndex (country, year).
    """
    from datetime import datetime
    countries  = countries  or THESIS_COUNTRIES
    indicators = indicators or WB_INDICATORS
    end_year   = end_year   or datetime.today().year - 1

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"market_moderators_{start_year}_{end_year}.csv"

    def _download():
        frames = []
        for wb_code, col_name in indicators.items():
            try:
                df = _fetch_wb(wb_code, col_name, countries, start_year, end_year)
                frames.append(df)
                print(f"  WB   {wb_code:<25s} ({col_name}) -> {len(df)} obs")
            except Exception as e:
                print(f"  [erro] WB {wb_code}: {e}")
        if not frames:
            raise RuntimeError("Nenhum indicador de mercado baixado.")
        return _merge_long(frames)

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 3. Consolidado - WGI + moderadores de mercado
# ---------------------------------------------------------------------------

def fetch_all_institutional(
    countries: list[str] | None = None,
    start_year: int = 2000,
    end_year: int | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Une WGI e moderadores de mercado em um único DataFrame anual.

    Retorna
    -------
    DataFrame com MultiIndex (country, year), pronto para linearmodels.panel.
    Colunas: Rule_of_Law, Regulatory_Quality, Govt_Effectiveness,
             Control_of_Corruption, Political_Stability, Voice_Accountability,
             MktCap_GDP_pct, Portfolio_Inflows_USD, GDP_USD

    Exemplo (painel com linearmodels)
    -----------------------------------
    from linearmodels.panel import PanelOLS

    inst = fetch_all_institutional()
    # inst já tem MultiIndex (country, year) - compatível com PanelOLS
    model = PanelOLS(y, X.join(inst), entity_effects=True, time_effects=True)
    """
    from datetime import datetime
    countries = countries or THESIS_COUNTRIES
    end_year  = end_year  or datetime.today().year - 1

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"institutional_all_{start_year}_{end_year}.csv"

    if not force and cache.exists():
        print(f"[cache] {cache.name}")
        return pd.read_csv(cache, index_col=["country", "year"])

    print(">> WGI (World Governance Indicators)...")
    wgi = fetch_wgi(
        countries=countries, start_year=start_year, end_year=end_year,
        data_dir=data_dir, force=force,
    )

    print(">> Moderadores de mercado (Cap/PIB, fluxos, PIB)...")
    mkt = fetch_market_moderators(
        countries=countries, start_year=start_year, end_year=end_year,
        data_dir=data_dir, force=force,
    )

    combined = wgi.join(mkt, how="outer").sort_index()

    # Adiciona nome legível do país como coluna auxiliar
    combined["country_name"] = combined.index.get_level_values("country").map(COUNTRY_NAMES)

    combined.to_csv(cache)
    print(f"\n>> Consolidado: {cache.name}  ({combined.shape[1]} colunas, {combined.shape[0]} obs)")

    # Resumo por país
    print(f"\n{'='*55}")
    print(f"  Cobertura por país")
    print(f"{'='*55}")
    for country in countries:
        try:
            sub = combined.loc[country]
            years = sub.index.tolist()
            name  = COUNTRY_NAMES.get(country, country)
            completeness = sub.notna().mean().mean()
            print(f"  {country}  {name:<14s}  {min(years)}-{max(years)}"
                  f"  completude: {completeness*100:.0f}%")
        except KeyError:
            print(f"  {country}  sem dados")

    return combined


# ---------------------------------------------------------------------------
# Exemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).parent)
    _DATA_DIR = str(Path(__file__).parent / "data" / "raw")

    inst = fetch_all_institutional(start_year=2000, data_dir=_DATA_DIR)

    print("\n--- Amostra: Brasil ---")
    print(inst.loc["BRA"].to_string())

    print("\n--- Corte transversal 2022 ---")
    print(inst.xs(2022, level="year")[["Regulatory_Quality", "MktCap_GDP_pct"]].to_string())
