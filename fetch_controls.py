"""
fetch_controls.py
-----------------
Coleta controles macrofinanceiros para os ensaios empíricos da tese.

Fontes:
  yfinance  : VIX, DXY
  FRED      : MOVE, DFF (Fed Funds), SOFR, SONIA, spread 10Y-2Y, EMBI
  URL pública: GPR - Caldara & Iacoviello (Excel)
               EPU - Baker, Bloom & Davis (Excel)
  Manual    : CDS soberano (Refinitiv/Markit - requer acesso pago)

FRED requer chave gratuita: https://fred.stlouisfed.org/docs/api/api_key.html
Salve em variável de ambiente:  FRED_API_KEY=sua_chave
Ou passe diretamente:           fetch_fred(api_key="sua_chave")

Cache: salva CSV em data_dir/; re-usa se já existir (force=True para re-baixar).

Dependências:
    pip install yfinance pandas requests openpyxl
    pip install fredapi   # para fetch_fred()

Uso em Jupyter:
    from fetch_controls import fetch_all_controls
    controls = fetch_all_controls(start="2015-01-01", fred_api_key="SUA_CHAVE")
"""

import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------

def _cached(path: Path, fn, force: bool) -> pd.DataFrame:
    if not force and path.exists():
        print(f"  [cache] {path.name}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fn()
    df.to_csv(path)
    print(f"  [salvo] {path.name}  ({df.shape[1]} colunas, {df.shape[0]} obs)")
    return df


# ---------------------------------------------------------------------------
# 1. yfinance - VIX, DXY
# ---------------------------------------------------------------------------

# Tickers usados em todos os ensaios
YFINANCE_TICKERS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
}


def fetch_yfinance(
    start: str = "2015-01-01",
    end: str | None = None,
    tickers: dict[str, str] | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Baixa VIX e DXY (ou qualquer ticker yfinance passado via `tickers`).

    Retorna
    -------
    DataFrame (Date × variável), fechamento diário
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    tickers = tickers or YFINANCE_TICKERS
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"controls_yf_{start}_{end}.csv"

    def _download():
        series = {}
        for name, tkr in tickers.items():
            try:
                raw = yf.Ticker(tkr).history(start=start, end=end, auto_adjust=True)
                if raw.empty:
                    print(f"  [sem dados] {tkr}")
                    continue
                s = raw["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                series[name] = s
            except Exception as e:
                print(f"  [erro] {tkr}: {e}")
            time.sleep(0.1)
        return pd.DataFrame(series).rename_axis("Date")

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 2. FRED - MOVE, DFF, SOFR, SONIA, spread 10Y-2Y, EMBI
# ---------------------------------------------------------------------------

# Séries da tabela de controles das notas de implementação
FRED_SERIES = {
    "DFF":    "DFF",      # Federal Funds Rate (diario)
    "SOFR":   "SOFR",     # Secured Overnight Financing Rate
    "SONIA":  "IUDSOIA",  # Sterling Overnight Index Average (Bank of England via FRED)
    "T10Y2Y": "T10Y2Y",   # Spread 10Y - 2Y Treasury (inclinacao da curva)
    # MOVE e EMBI nao disponiveis no FRED publico (requerem Bloomberg/Refinitiv)
}


def fetch_fred(
    series: dict[str, str] | list[str] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    api_key: str | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Baixa séries do FRED via fredapi.

    Parâmetros
    ----------
    series  : dict {nome_coluna: fred_id} ou lista de fred_ids
              None -> baixa FRED_SERIES completo
    api_key : chave FRED; se None, lê FRED_API_KEY do ambiente

    Retorna
    -------
    DataFrame (Date × série), diário, forward-fill aplicado

    Exemplo
    -------
    df = fetch_fred({"dff": "DFF", "move": "MOVE"}, api_key="abc123")
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Instale fredapi: pip install fredapi")

    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "Chave FRED não encontrada.\n"
            "  Opção 1: fetch_fred(api_key='SUA_CHAVE')\n"
            "  Opção 2: defina FRED_API_KEY no ambiente\n"
            "  Chave gratuita: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    end = end or datetime.today().strftime("%Y-%m-%d")

    if series is None:
        series = FRED_SERIES
    elif isinstance(series, list):
        series = {s: s for s in series}

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    tag = "_".join(sorted(series.values()))[:40]
    cache = Path(data_dir) / f"controls_fred_{tag}_{start}_{end}.csv"

    def _download():
        fred = Fred(api_key=api_key)
        frames = {}
        for col, fred_id in series.items():
            try:
                s = fred.get_series(fred_id, observation_start=start, observation_end=end)
                s.index = pd.to_datetime(s.index)
                s.name = col
                frames[col] = s
                print(f"  FRED  {fred_id:<12s} -> {len(s)} obs")
            except Exception as e:
                print(f"  [erro] FRED {fred_id}: {e}")
        df = pd.DataFrame(frames).rename_axis("Date")
        return df.resample("D").last().ffill()

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 3. GPR - Geopolitical Risk Index (Caldara & Iacoviello, 2022)
# ---------------------------------------------------------------------------

_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xlsx"


def fetch_gpr(
    freq: str = "monthly",
    start: str = "2015-01-01",
    end: str | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Baixa o Geopolitical Risk Index (Caldara & Iacoviello, 2022).
    Fonte: https://www.matteoiacoviello.com/gpr.htm

    Colunas: GPR (índice agregado), GPRA (ameaças), GPRT (eventos reais),
             + índices por país (quando disponíveis).

    Parâmetros
    ----------
    freq : "monthly" (default) | "daily"
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"gpr_{freq}_{start}_{end}.csv"

    def _download():
        import requests, io
        print(f"  Baixando GPR ({freq}) de Caldara & Iacoviello...")
        r = requests.get(_GPR_URL, headers={"User-Agent": "research/1.0"}, timeout=30)
        r.raise_for_status()

        xl = pd.ExcelFile(io.BytesIO(r.content))

        # Mapa de frequência -> nome esperado da aba (o arquivo usa nomes fixos, não "monthly"/"daily")
        _SHEET_PREF = {
            "monthly": ["GPR", "monthly", "Monthly", "data"],
            "daily":   ["GPR_HISTORICAL", "daily", "Daily", "GPR"],
        }
        preferred = _SHEET_PREF.get(freq, ["GPR"])
        sheet = next(
            (s for s in xl.sheet_names for p in preferred if s.upper() == p.upper()),
            next((s for s in xl.sheet_names if "important" not in s.lower()), xl.sheet_names[0]),
        )
        print(f"  Aba selecionada: '{sheet}'  (disponíveis: {xl.sheet_names})")

        # Incrementa skiprows até os nomes de coluna conterem "year"/"month" ou "gpr"
        df = None
        for skip in range(20):
            candidate = xl.parse(sheet, skiprows=skip)
            cols = [str(c).lower().strip() for c in candidate.columns]
            if ("year" in cols and "month" in cols) or any(c == "gpr" for c in cols):
                df = candidate
                break

        if df is None:
            raise ValueError(
                f"Cabeçalho dos dados GPR não encontrado na aba '{sheet}' "
                f"nas primeiras 20 linhas.\nAbas disponíveis: {xl.sheet_names}"
            )

        df.columns = [str(c).strip() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}

        if "year" in col_map and "month" in col_map:
            year_col  = col_map["year"]
            month_col = col_map["month"]
            # Descarta linhas de rodapé (notas de texto onde year/month não são numéricos)
            df[year_col]  = pd.to_numeric(df[year_col],  errors="coerce")
            df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
            df = df.dropna(subset=[year_col, month_col])
            df[year_col]  = df[year_col].astype(int)
            df[month_col] = df[month_col].astype(int)
            df.index = pd.to_datetime({
                "year": df[year_col], "month": df[month_col], "day": 1
            })
            df = df.drop(columns=[year_col, month_col], errors="ignore")
        else:
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if date_col is None:
                raise ValueError(
                    f"Coluna de data não encontrada. Colunas: {df.columns.tolist()}"
                )
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df.index = df[date_col]
            df = df.drop(columns=[date_col])

        df.index.name = "Date"
        # Remove colunas sem nome, inteiramente vazias e coluna interna n11
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.dropna(axis=1, how="all")
        df = df.drop(columns=["n11"], errors="ignore")
        return df.sort_index().loc[start:end]

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 4. EMBI - Emerging Market Bond Index (arquivo local JP Morgan / IADB)
# ---------------------------------------------------------------------------

# Mapeamento coluna Excel -> nome padronizado
_EMBI_COL_MAP = {
    "Global":      "EMBI_Global",
    "LATINO":      "EMBI_LATINO",
    "Brasil":      "EMBI_BRA",
    "Mexico":      "EMBI_MEX",
    "Argentina":   "EMBI_ARG",
    "Colombia":    "EMBI_COL",
    "Peru":        "EMBI_PER",
    "Panama":      "EMBI_PAN",
    "Uruguay":     "EMBI_URY",
    "Venezuela":   "EMBI_VEN",
    "Chile":       "EMBI_CHL",
    "Ecuador":     "EMBI_ECU",
}


def fetch_embi_local(
    filepath: str = "Serie_Historica_Spread_del_EMBI.xlsx",
    start: str = "2015-01-01",
    end: str | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Carrega o EMBI Global Diversified de arquivo Excel local (JP Morgan / IADB).

    O arquivo deve estar na mesma pasta do script ou caminho absoluto.
    Colunas retornadas: EMBI_Global, EMBI_BRA, EMBI_MEX, EMBI_LATINO, ...
    Valores em decimais (ex: 2.10 = spread de 210 bps).

    Parametros
    ----------
    filepath : caminho para o .xlsx (relativo ao cwd ou absoluto)
    start    : data inicial
    end      : data final
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"embi_local_{start}_{end}.csv"

    def _download():
        path = Path(filepath)
        # Se caminho relativo, resolve a partir da pasta deste script
        if not path.is_absolute():
            path = Path(__file__).parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo EMBI nao encontrado: {path.resolve()}\n"
                f"Coloque o arquivo na pasta do projeto."
            )

        # Cabecalho real esta na segunda linha do Excel (header=1)
        df = pd.read_excel(path, header=1)

        # Primeira coluna e a data
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df.index = df[date_col]
        df.index.name = "Date"
        df = df.drop(columns=[date_col])

        # Normaliza nomes de colunas (remove acentos simples para match)
        col_norm = {c: c.strip() for c in df.columns}
        # Trata variacoes de nome (ex: "M?xico" vs "Mexico")
        col_norm_lower = {c.lower().replace("é","e").replace("ú","u")
                          .replace("á","a").replace("ó","o"): c
                          for c in df.columns}

        rename = {}
        for raw_name, std_name in _EMBI_COL_MAP.items():
            key = raw_name.lower().replace("é","e").replace("ú","u")
            if key in col_norm_lower:
                rename[col_norm_lower[key]] = std_name

        df = df.rename(columns=rename)
        # Mantém apenas as colunas mapeadas
        keep = [c for c in df.columns if c.startswith("EMBI_")]
        df = df[keep]

        # Converte para numerico (alguns campos podem ter texto)
        df = df.apply(pd.to_numeric, errors="coerce")

        print(f"  EMBI local: {df.shape[1]} series | "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        return df.sort_index().loc[start:end]

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 5. EPU - Economic Policy Uncertainty (Baker, Bloom & Davis, 2016)
# ---------------------------------------------------------------------------

_EPU_URLS = {
    "US":     "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx",
    "global": "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx",
}


def fetch_epu(
    country: str = "US",
    start: str = "2015-01-01",
    end: str | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Baixa o Economic Policy Uncertainty Index (Baker, Bloom & Davis, 2016).
    Fonte: https://www.policyuncertainty.com/
    Frequência: mensal.

    Parâmetros
    ----------
    country : "US" | "global"
    """
    if country not in _EPU_URLS:
        raise ValueError(f"country deve ser {list(_EPU_URLS)}. Recebido: '{country}'")

    end = end or datetime.today().strftime("%Y-%m-%d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"epu_{country}_{start}_{end}.csv"

    def _download():
        import requests, io
        url = _EPU_URLS[country]
        print(f"  Baixando EPU ({country}) de Baker, Bloom & Davis...")
        r = requests.get(url, headers={"User-Agent": "research/1.0"}, timeout=30)
        r.raise_for_status()

        df = pd.read_excel(io.BytesIO(r.content))

        # Normaliza colunas para case-insensitive
        col_map = {str(c).lower().strip(): c for c in df.columns}
        if "year" not in col_map or "month" not in col_map:
            raise ValueError(f"Colunas Year/Month não encontradas. Colunas: {df.columns.tolist()}")

        year_col  = col_map["year"]
        month_col = col_map["month"]

        # Descarta linhas de rodapé (notas de texto onde Year/Month não são numéricos)
        df[year_col]  = pd.to_numeric(df[year_col],  errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        df = df.dropna(subset=[year_col, month_col])
        df[year_col]  = df[year_col].astype(int)
        df[month_col] = df[month_col].astype(int)

        df.index = pd.to_datetime({"year": df[year_col], "month": df[month_col], "day": 1})
        df.index.name = "Date"
        df = df.drop(columns=[year_col, month_col]).sort_index()
        return df.loc[start:end]

    return _cached(cache, _download, force)


# ---------------------------------------------------------------------------
# 5. Consolidado - todos os controles em um DataFrame
# ---------------------------------------------------------------------------

def fetch_all_controls(
    start: str = "2015-01-01",
    end: str | None = None,
    fred_api_key: str | None = None,
    fred_series: dict[str, str] | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Coleta e une todos os controles disponíveis em um único DataFrame diário.
    FRED é incluído apenas se api_key estiver disponível.
    GPR e EPU têm frequência mensal - são alinhados para diário via forward-fill.

    Parâmetros
    ----------
    fred_api_key : chave FRED (ou None para pular séries FRED)
    fred_series  : dict customizado para fetch_fred(); None -> usa FRED_SERIES
    data_dir     : pasta de cache
    force        : re-baixa tudo

    Retorna
    -------
    DataFrame (Date × controle), diário, sem NaN no início de cada série
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    frames = []

    print(">> yfinance (VIX, DXY)...")
    frames.append(fetch_yfinance(start=start, end=end, data_dir=data_dir, force=force))

    key = fred_api_key or os.getenv("FRED_API_KEY")
    if key:
        print(">> FRED (MOVE, DFF, SOFR, SONIA, T10Y2Y, EMBI)...")
        try:
            frames.append(fetch_fred(
                series=fred_series,
                start=start, end=end,
                api_key=key,
                data_dir=data_dir, force=force,
            ))
        except Exception as e:
            print(f"  [AVISO] FRED: {e}")
    else:
        print(">> FRED ignorado (FRED_API_KEY não encontrada).")

    print(">> GPR (Caldara & Iacoviello)...")
    try:
        frames.append(fetch_gpr(start=start, end=end, data_dir=data_dir, force=force))
    except Exception as e:
        print(f"  [AVISO] GPR: {e}")

    print(">> EPU (Baker, Bloom & Davis)...")
    try:
        frames.append(fetch_epu(start=start, end=end, data_dir=data_dir, force=force))
    except Exception as e:
        print(f"  [AVISO] EPU: {e}")

    print(">> EMBI (arquivo local)...")
    try:
        frames.append(fetch_embi_local(start=start, end=end, data_dir=data_dir, force=force))
    except Exception as e:
        print(f"  [AVISO] EMBI: {e}")

    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.resample("D").last().ffill().loc[start:end]

    out = Path(data_dir) / f"controls_all_{start}_{end}.csv"
    combined.to_csv(out)
    print(f"\n>> Consolidado: {out.name}  ({combined.shape[1]} variáveis, {combined.shape[0]} obs)")
    return combined


# ---------------------------------------------------------------------------
# Exemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).parent)
    _DATA_DIR = str(Path(__file__).parent / "data" / "raw")

    controls = fetch_all_controls(
        start="2015-01-01",
        data_dir=_DATA_DIR,
        fred_api_key=os.environ.get("FRED_API_KEY", ""),
    )
    print(controls.tail())
    print(controls.info())

    # Com FRED:
    # controls = fetch_all_controls(start="2015-01-01", fred_api_key="SUA_CHAVE")
