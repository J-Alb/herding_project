"""
fetch_market.py
---------------
Funções para coleta de dados de mercado via yfinance.

Funções principais:
    fetch_asset(ticker, start, end, interval)         -> dados OHLCV + retorno log de um ativo
    fetch_sp500_top_n(n, start, end)                  -> top-N empresas do S&P 500 por market cap
    compute_portfolios(prices_df, meta_df)            -> retornos EW e MCW dos portfolios

Uso:
    python fetch_market.py

Dependências:
    pip install yfinance pandas numpy matplotlib
"""

import time
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. FUNÇÃO CORE - ATIVO INDIVIDUAL
# ---------------------------------------------------------------------------

def fetch_asset(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Baixa série histórica de um único ativo via yfinance.

    Parâmetros
    ----------
    ticker   : str  - símbolo do ativo (ex: 'AAPL', '^GSPC', 'PETR4.SA')
    start    : str  - data inicial no formato 'YYYY-MM-DD'
    end      : str  - data final  no formato 'YYYY-MM-DD' (default: hoje)
    interval : str  - granularidade: '1d', '1wk', '1mo'
    verbose  : bool - imprime resumo ao final

    Retorna
    -------
    pd.DataFrame com colunas:
        Open, High, Low, Close, Adj_Close, Volume,
        Log_Return, Abs_Return, Cumulative_Return
    """
    end = end or datetime.today().strftime("%Y-%m-%d")

    tkr = yf.Ticker(ticker)
    raw = tkr.history(start=start, end=end, interval=interval, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"Nenhum dado retornado para '{ticker}'. Verifique o ticker.")

    # Padroniza colunas
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["Open", "High", "Low", "Adj_Close", "Volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)  # remove tz-aware
    df.index.name = "Date"

    # Retornos
    df["Log_Return"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(1))
    df["Abs_Return"] = df["Adj_Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Abs_Return"]).cumprod() - 1

    # Market cap instantâneo (se disponível)
    info = tkr.fast_info
    mktcap = getattr(info, "market_cap", None)
    if mktcap:
        df.attrs["market_cap"] = mktcap
        df.attrs["currency"] = getattr(info, "currency", "USD")

    df.attrs["ticker"] = ticker
    df.dropna(subset=["Log_Return"], inplace=True)

    if verbose:
        print(f"[fetch_asset] {ticker:>10s} | "
              f"{df.index[0].date()} -> {df.index[-1].date()} | "
              f"{len(df):>5d} obs | "
              f"Adj_Close: {df['Adj_Close'].iloc[-1]:.2f} | "
              f"Retorno total: {df['Cumulative_Return'].iloc[-1]*100:.1f}%")

    return df


# ---------------------------------------------------------------------------
# 2. HELPER - OBTÉM LISTA DE TICKERS DO S&P 500 (via Wikipedia)
# ---------------------------------------------------------------------------

def _get_sp500_tickers() -> pd.DataFrame:
    """
    Lê a lista atual de constituintes do S&P 500 da Wikipedia.
    Retorna DataFrame com colunas: Symbol, Security, GICS_Sector, GICS_SubIndustry
    """
    import requests
    from bs4 import BeautifulSoup

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (research script)"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Não foi possível acessar a Wikipedia: {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})

    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) >= 4:
            rows.append(
                {
                    "Symbol": cols[0].replace(".", "-"),  # BRK.B -> BRK-B
                    "Security": cols[1],
                    "GICS_Sector": cols[2],
                    "GICS_SubIndustry": cols[3],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. FUNÇÃO PRINCIPAL - TOP-N DO S&P 500 POR MARKET CAP
# ---------------------------------------------------------------------------

def fetch_sp500_top_n(
    n: int = 30,
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    sleep_sec: float = 0.25,
    save_csv: bool = True,
    csv_prefix: str = "sp500_top",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna as N maiores empresas do S&P 500 por market cap atual,
    com preços ajustados e market cap para cada uma.

    Parâmetros
    ----------
    n          : número de empresas a retornar (default: 30)
    start/end  : janela histórica de preços
    interval   : granularidade dos preços
    sleep_sec  : pausa entre requests (respeitar rate limit do Yahoo)
    save_csv   : salva resultados em CSV
    csv_prefix : prefixo dos arquivos CSV

    Retorna
    -------
    prices_df  : DataFrame wide (Date × Ticker) com preços ajustados
    meta_df    : DataFrame com metadados das N empresas (ticker, nome, setor, market cap)
    """
    end = end or datetime.today().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Coletando top-{n} empresas do S&P 500 por market cap")
    print(f"  Janela: {start} -> {end}  |  Intervalo: {interval}")
    print(f"{'='*60}\n")

    # Passo 1: lista de constituintes
    print(">> Obtendo lista de constituintes do S&P 500 (Wikipedia)...")
    sp500_df = _get_sp500_tickers()
    tickers = sp500_df["Symbol"].tolist()
    print(f"  {len(tickers)} empresas encontradas.\n")

    # Passo 2: coleta de market cap via yfinance.fast_info
    print(f">> Coletando market cap de {len(tickers)} empresas...")
    mcap_records = []

    for i, tkr_str in enumerate(tickers, 1):
        try:
            info = yf.Ticker(tkr_str).fast_info
            mktcap = getattr(info, "market_cap", None)
            if mktcap and mktcap > 0:
                mcap_records.append(
                    {
                        "Symbol": tkr_str,
                        "Market_Cap_USD": mktcap,
                        "Currency": getattr(info, "currency", "USD"),
                    }
                )
        except Exception:
            pass  # ignora falhas individuais

        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} processados...")
        time.sleep(sleep_sec)

    mcap_df = pd.DataFrame(mcap_records).sort_values("Market_Cap_USD", ascending=False)

    # Merge com metadados de setor
    mcap_df = mcap_df.merge(
        sp500_df.rename(columns={"Symbol": "Symbol"}),
        on="Symbol",
        how="left",
    )

    top_n = mcap_df.head(n).reset_index(drop=True)
    top_n["Rank"] = range(1, n + 1)

    # Formata market cap em trilhões/bilhões para leitura
    top_n["Market_Cap_fmt"] = top_n["Market_Cap_USD"].apply(
        lambda x: f"${x/1e12:.2f}T" if x >= 1e12 else f"${x/1e9:.1f}B"
    )

    print(f"\n  Top-{n} por market cap identificado.\n")

    # Passo 3: coleta de séries de preço ajustado
    print(">> Baixando séries históricas de preços...")
    price_dict = {}

    for _, row in top_n.iterrows():
        tkr_str = row["Symbol"]
        try:
            df = fetch_asset(tkr_str, start=start, end=end, interval=interval)
            price_dict[tkr_str] = df["Adj_Close"]
        except Exception as e:
            print(f"  [AVISO] {tkr_str}: {e}")
        time.sleep(sleep_sec)

    prices_df = pd.DataFrame(price_dict)
    prices_df.index.name = "Date"

    # Passo 4: retornos logarítmicos (wide format)
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

    # Salva CSVs
    if save_csv:
        meta_file = f"{csv_prefix}{n}_metadata.csv"
        prices_file = f"{csv_prefix}{n}_prices.csv"
        returns_file = f"{csv_prefix}{n}_log_returns.csv"

        top_n.to_csv(meta_file, index=False)
        prices_df.to_csv(prices_file)
        returns_df.to_csv(returns_file)

        print(f"\n  Arquivos salvos:")
        print(f"    {meta_file}")
        print(f"    {prices_file}")
        print(f"    {returns_file}")

    # Resumo
    print(f"\n{'='*60}")
    print(f"  RESUMO - Top-{n} S&P 500 por Market Cap")
    print(f"{'='*60}")
    for _, row in top_n[["Rank", "Symbol", "Security", "GICS_Sector", "Market_Cap_fmt"]].iterrows():
        print(
            f"  #{row['Rank']:>2d}  {row['Symbol']:<8s}  {row['Security']:<35s}"
            f"  {row['Market_Cap_fmt']:>9s}  [{row['GICS_Sector']}]"
        )

    return prices_df, top_n


# ---------------------------------------------------------------------------
# 4. PORTFOLIOS - EQUAL-WEIGHTED E MARKET-CAP-WEIGHTED
# ---------------------------------------------------------------------------

def compute_portfolios(
    prices_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    save_csv: bool = True,
    csv_prefix: str = "sp500_top",
) -> pd.DataFrame:
    """
    Calcula retornos de portfolios equal-weighted e market-cap-weighted.

    Parâmetros
    ----------
    prices_df : DataFrame wide (Date × Ticker) com preços ajustados
    meta_df   : DataFrame com colunas Symbol e Market_Cap_USD
    save_csv  : salva resultados em CSV

    Retorna
    -------
    DataFrame com colunas:
        EW_Return, EW_Cumulative,
        MCW_Return, MCW_Cumulative
    """
    # Alinha tickers disponíveis em ambos os DataFrames
    valid = [t for t in meta_df["Symbol"] if t in prices_df.columns]
    prices = prices_df[valid].copy()

    log_ret = np.log(prices / prices.shift(1)).dropna()

    # Pesos market-cap (normalizados para somar 1)
    mcap = meta_df.set_index("Symbol").loc[valid, "Market_Cap_USD"]
    mcap_weights = (mcap / mcap.sum()).values  # shape (N,)

    # Equal-weighted
    ew_ret = log_ret.mean(axis=1)

    # Market-cap-weighted
    mcw_ret = pd.Series(
        log_ret.values @ mcap_weights,
        index=log_ret.index,
        name="MCW_Return",
    )

    port = pd.DataFrame(
        {"EW_Return": ew_ret, "MCW_Return": mcw_ret},
        index=log_ret.index,
    )

    # Retornos acumulados (em termos de retorno simples acumulado)
    port["EW_Cumulative"] = np.exp(port["EW_Return"].cumsum()) - 1
    port["MCW_Cumulative"] = np.exp(port["MCW_Return"].cumsum()) - 1

    n = len(valid)
    print(f"\n[compute_portfolios] {n} ativos | "
          f"{port.index[0].date()} -> {port.index[-1].date()}")
    print(f"  EW  retorno total : {port['EW_Cumulative'].iloc[-1]*100:+.1f}%")
    print(f"  MCW retorno total : {port['MCW_Cumulative'].iloc[-1]*100:+.1f}%")

    if save_csv:
        fname = f"{csv_prefix}{n}_portfolios.csv"
        port.to_csv(fname)
        print(f"  Arquivo salvo: {fname}")

    return port


# ---------------------------------------------------------------------------
# 5. VISUALIZAÇÕES
# ---------------------------------------------------------------------------

def plot_prices(prices_df: pd.DataFrame, meta_df: pd.DataFrame, top_k: int = 10) -> None:
    """Plota preços normalizados (base 100) das top-k empresas."""
    top_tickers = meta_df.head(top_k)["Symbol"].tolist()
    subset = prices_df[top_tickers].dropna()
    normalized = subset / subset.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    for tkr in top_tickers:
        ax.plot(normalized.index, normalized[tkr], linewidth=1.2, label=tkr)

    ax.set_title(f"Preços Normalizados - Top-{top_k} S&P 500 (base 100)", fontsize=14)
    ax.set_xlabel("Data")
    ax.set_ylabel("Índice de preço (base 100)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"sp500_top{top_k}_prices_normalized.png", dpi=150)
    plt.show()
    print("  Gráfico salvo: sp500_top{top_k}_prices_normalized.png")


def plot_market_cap(meta_df: pd.DataFrame, top_k: int = 30) -> None:
    """Plota barras horizontais com market cap das top-k empresas."""
    subset = meta_df.head(top_k).copy()
    subset["Market_Cap_T"] = subset["Market_Cap_USD"] / 1e12

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.85, len(subset)))
    bars = ax.barh(
        subset["Symbol"][::-1],
        subset["Market_Cap_T"][::-1],
        color=colors,
    )

    # Anotações
    for bar, (_, row) in zip(bars, subset[::-1].iterrows()):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            row["Market_Cap_fmt"],
            va="center",
            fontsize=8,
        )

    ax.set_title(f"Top-{top_k} S&P 500 - Market Cap (USD Trilhões)", fontsize=13)
    ax.set_xlabel("Market Cap (USD Tri)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fT"))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"sp500_top{top_k}_marketcap.png", dpi=150)
    plt.show()
    print(f"  Gráfico salvo: sp500_top{top_k}_marketcap.png")


def plot_return_heatmap(returns_df: pd.DataFrame, meta_df: pd.DataFrame, top_k: int = 20) -> None:
    """Heatmap de correlação entre retornos das top-k empresas."""
    top_tickers = meta_df.head(top_k)["Symbol"].tolist()
    corr = returns_df[top_tickers].dropna().corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlação de Pearson")

    ax.set_xticks(range(len(top_tickers)))
    ax.set_yticks(range(len(top_tickers)))
    ax.set_xticklabels(top_tickers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_tickers, fontsize=8)
    ax.set_title(f"Correlação de Retornos Logarítmicos - Top-{top_k} S&P 500", fontsize=13)

    # Valores na célula
    for i in range(len(top_tickers)):
        for j in range(len(top_tickers)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    plt.savefig(f"sp500_top{top_k}_correlation.png", dpi=150)
    plt.show()
    print(f"  Gráfico salvo: sp500_top{top_k}_correlation.png")


def plot_portfolios(port: pd.DataFrame, top_k: int | None = None) -> None:
    """Plota retorno acumulado dos portfolios EW e MCW."""
    label = f"Top-{top_k} S&P 500" if top_k else "S&P 500"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Retorno acumulado
    ax = axes[0]
    ax.plot(port.index, port["EW_Cumulative"] * 100, label="Equal-Weighted", linewidth=1.5)
    ax.plot(port.index, port["MCW_Cumulative"] * 100, label="Market-Cap-Weighted", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(f"Retorno Acumulado dos Portfolios - {label}", fontsize=13)
    ax.set_ylabel("Retorno acumulado (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Retornos diários
    ax = axes[1]
    ax.plot(port.index, port["EW_Return"] * 100, label="EW diário", linewidth=0.8, alpha=0.7)
    ax.plot(port.index, port["MCW_Return"] * 100, label="MCW diário", linewidth=0.8, alpha=0.7, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("Retornos Log-Diários", fontsize=13)
    ax.set_ylabel("Retorno log (%)")
    ax.set_xlabel("Data")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    suffix = f"top{top_k}" if top_k else "portfolio"
    fname = f"sp500_{suffix}_portfolios.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Gráfico salvo: {fname}")


# ---------------------------------------------------------------------------
# 6. COMPONENTES POR MERCADO - OS 8 MERCADOS DA TESE
# ---------------------------------------------------------------------------

_MARKETS = {
    "sp500":     {"index": "^GSPC",     "name": "S&P 500"},
    "ftse100":   {"index": "^FTSE",     "name": "FTSE 100"},
    "dax40":     {"index": "^GDAXI",    "name": "DAX 40"},
    "nikkei225": {"index": "^N225",     "name": "Nikkei 225"},
    "ibovespa":  {"index": "^BVSP",     "name": "Ibovespa"},
    "ipc":       {"index": "^MXX",      "name": "IPC México"},
    "csi300":    {"index": "000300.SS", "name": "CSI 300"},
    "nifty50":   {"index": "^NSEI",     "name": "Nifty 50"},
}


def _get_tickers(market: str) -> list[str]:
    """
    Obtém tickers dos componentes de um índice via fontes abertas.

    Suporte automático: sp500, ftse100, dax40, nikkei225, ibovespa, nifty50
    Para ipc e csi300: passe os tickers manualmente em load_or_fetch_market().
    """
    import requests
    from io import StringIO

    def _wiki_col(url: str, col_substr: str, suffix: str = "") -> list[str]:
        r = requests.get(url, headers={"User-Agent": "research/1.0"}, timeout=15)
        r.raise_for_status()
        for t in pd.read_html(StringIO(r.text)):
            match = [c for c in t.columns if col_substr.lower() in str(c).lower()]
            if match:
                tks = t[match[0]].dropna().astype(str).str.strip()
                tks = tks[~tks.isin(["nan", ""])]
                return (tks + suffix).tolist()
        raise ValueError(f"Coluna '{col_substr}' não encontrada em {url}")

    if market == "sp500":
        return _get_sp500_tickers()["Symbol"].tolist()

    if market == "ftse100":
        # Wikipedia renamed the column from "EPIC" to "Ticker"
        return _wiki_col("https://en.wikipedia.org/wiki/FTSE_100_Index", "Ticker", ".L")

    if market == "dax40":
        # Wikipedia table already includes ".DE" in the ticker — no suffix needed
        return _wiki_col("https://en.wikipedia.org/wiki/DAX", "Ticker", "")

    if market == "nikkei225":
        # Wikipedia has no parseable ticker table for Nikkei 225.
        # Uses a pre-extracted JSON built from Wikipedia company infoboxes (TYO codes).
        import json as _json
        _json_path = Path(__file__).parent / "_nikkei225_tickers.json"
        if not _json_path.exists():
            raise FileNotFoundError(
                "_nikkei225_tickers.json not found. "
                "Run _fetch_nikkei_tickers.py once to generate it."
            )
        with open(_json_path) as f:
            return _json.load(f)

    if market == "nifty50":
        return _wiki_col("https://en.wikipedia.org/wiki/NIFTY_50", "Symbol", ".NS")

    if market == "ibovespa":
        import base64, json
        params = {"language": "pt-br", "pageNumber": 1, "pageSize": 120,
                  "index": "IBOV", "segment": "1"}
        encoded = base64.b64encode(json.dumps(params).encode()).decode()
        url = (f"https://sistemaswebb3-listados.b3.com.br/indexProxy/"
               f"indexCall/GetPortfolioDay/{encoded}")
        r = requests.get(url, headers={"User-Agent": "research/1.0"}, timeout=15)
        r.raise_for_status()
        return [item["cod"] + ".SA" for item in r.json().get("results", [])]

    raise NotImplementedError(
        f"Auto-fetch não disponível para '{market}'. "
        f"Passe os tickers manualmente: "
        f"load_or_fetch_market('{market}', tickers=[...])"
    )


def load_or_fetch_market(
    market: str,
    start: str = "2015-01-01",
    end: str | None = None,
    tickers: list[str] | None = None,
    data_dir: str = "./data/raw",
    force: bool = False,
) -> pd.DataFrame:
    """
    Retorna preços de fechamento ajustados dos componentes de um índice.
    Carrega do disco se o arquivo já existir; baixa e salva caso contrário.

    Parâmetros
    ----------
    market   : "sp500" | "ftse100" | "dax40" | "nikkei225" |
               "ibovespa" | "nifty50" | "ipc"* | "csi300"*
               (* requer tickers manuais)
    start    : data inicial  (YYYY-MM-DD)
    end      : data final    (YYYY-MM-DD, default: hoje)
    tickers  : lista customizada; se None, busca automaticamente
    data_dir : pasta para salvar/carregar o CSV
    force    : True -> re-baixa mesmo se o arquivo já existir

    Retorna
    -------
    DataFrame wide - Date × Ticker (preços ajustados)

    Exemplo (Jupyter)
    -----------------
    from fetch_market import load_or_fetch_market
    import numpy as np

    prices = load_or_fetch_market("ibovespa", start="2018-01-01")
    returns = np.log(prices / prices.shift(1)).dropna()
    """
    if market not in _MARKETS:
        raise ValueError(f"Mercado desconhecido: '{market}'. Opções: {list(_MARKETS)}")

    end = end or datetime.today().strftime("%Y-%m-%d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(data_dir) / f"{market}_{start}_{end}.csv"
    name  = _MARKETS[market]["name"]

    if not force and cache.exists():
        print(f"[{name}] carregando cache: {cache.name}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    if tickers is None:
        print(f"[{name}] obtendo componentes...")
        tickers = _get_tickers(market)
        print(f"  {len(tickers)} componentes.")

    print(f"[{name}] baixando {len(tickers)} tickers | {start} -> {end}")
    series: dict[str, pd.Series] = {}
    for tkr in tickers:
        try:
            raw = yf.Ticker(tkr).history(start=start, end=end, auto_adjust=True)
            if raw.empty:
                continue
            s = raw["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series[tkr] = s
        except Exception:
            pass
        time.sleep(0.15)

    df = pd.DataFrame(series)
    df.index.name = "Date"
    df.to_csv(cache)
    print(f"  salvo: {cache.name}  ({df.shape[1]} ativos, {df.shape[0]} obs)")
    return df


# ---------------------------------------------------------------------------
# 7. EXECUÇÃO DIRETA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).parent)

    # --- Exemplo 1: ativo individual ---
    print("\n" + "="*60)
    print("  EXEMPLO 1 - Ativo individual: AAPL")
    print("="*60)
    aapl = fetch_asset("AAPL", start="2020-01-01", end="2025-12-31")
    print(aapl.tail(3).to_string())
    print(f"\n  Market cap atual: ${aapl.attrs.get('market_cap', 'N/A'):,}")

    # --- Exemplo 2: top-30 do S&P 500 ---
    prices, meta = fetch_sp500_top_n(
        n=30,
        start="2020-01-01",
        end="2025-12-31",
        interval="1d",
        sleep_sec=0.3,
        save_csv=True,
    )

    # --- Portfolios ---
    print("\n Calculando portfolios EW e MCW...")
    port = compute_portfolios(prices, meta, save_csv=True, csv_prefix="sp500_top")

    # --- Visualizações ---
    print("\n Gerando visualizações...")
    plot_market_cap(meta, top_k=30)
    plot_prices(prices, meta, top_k=10)
    plot_return_heatmap(
        np.log(prices / prices.shift(1)).dropna(),
        meta,
        top_k=20,
    )
    plot_portfolios(port, top_k=30)

    print("\nOK Execução concluída.")
