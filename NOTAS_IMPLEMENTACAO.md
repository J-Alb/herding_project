# Notas de Implementação — Tese em Três Ensaios
## Herding behavior, reflexividade e dinâmica de preços em mercados globais de equities

**Autor:** Giovani Silva  
**Atualizado:** 31 Maio/2026  
**Stack principal:** Python 3.14, pandas, numpy, yfinance, statsmodels, sklearn, transformers

---

## Arquivos implementados (estado atual)

```
herding_reflexivity_prices/
├── data/
│   └── raw/                             # cache local de todos os dados
│       ├── sp500_*.csv                  # preços S&P 500 (503 ativos)
│       ├── ibovespa_*.csv               # preços Ibovespa (79 ativos)
│       ├── nifty50_*.csv                # preços Nifty 50 (50 ativos)
│       ├── ftse100_*.csv                # preços FTSE 100 (100 ativos)
│       ├── dax40_*.csv                  # preços DAX 40 (40 ativos, 36 válidos)
│       ├── nikkei225_*.csv              # preços Nikkei 225 (167 ativos via Wikipedia API)
│       ├── csad_{market}_*.csv          # CSAD + Rm + abs_Rm + Rm_sq + D_up/down + N
│       ├── ar_{market}_w63_k1.csv       # PC1-share rolling (janela 63d)
│       ├── cck_all_markets.csv          # CCK beta2 full vs sub-sample, 6 mercados
│       ├── controls_all_*.csv           # 29 controles macro diários
│       ├── controls_yf_*.csv            # VIX, DXY
│       ├── controls_fred_*.csv          # DFF, SOFR, SONIA, T10Y2Y
│       ├── gpr_monthly_*.csv            # GPR (Caldara & Iacoviello)
│       ├── epu_US_*.csv                 # EPU (Baker, Bloom & Davis)
│       ├── embi_local_*.csv             # EMBI (10 países, 2007-2026)
│       ├── wgi_*.csv                    # WGI (6 indicadores, 8 países)
│       ├── market_moderators_*.csv      # Cap/PIB, fluxos, PIB
│       └── institutional_all_*.csv      # consolidado moderadores
│
├── _nikkei225_tickers.json  # 167 tickers Nikkei via Wikipedia API (TYO|XXXX)
├── _fetch_nikkei_tickers.py # script para re-extrair tickers Nikkei
│
├── fetch_market.py          ✅ 6 mercados funcionando + filtro cobertura + ffill
├── fetch_controls.py        ✅ VIX, DXY, FRED, GPR, EPU, EMBI local
├── fetch_institutional.py   ✅ WGI, Cap/PIB (World Bank REST API)
├── csad.py                  ✅ CSAD + diagnóstico + multi-mercado
├── absorption_ratio.py      ✅ PC1-share rolling (janela configurável)
├── cck_symmetric.py         ✅ CCK simétrico + assimétrico (OLS NW)
├── cck_quantile.py          ✅ CCK quantile simétrico + assimétrico
├── 00_exemplo_coleta.py     ✅ exemplo coleta completa
├── 01_exemplo_csad.py       ✅ CSAD plots + CCK SP500 e Ibovespa
├── 02_subsample_analysis.py ✅ full vs sub-sample + PC1-share dual axis
├── 03_all_markets.py        ✅ 6 mercados: estatísticas, CSAD, PC1-share, CCK
├── absorption_ratio_notes.tex  ✅ derivação matemática completa (Overleaf)
└── pdfs/
    ├── presentation.tex     ✅ 14 slides Beamer (6 mercados, CCK tabela, plots)
    └── absorption_ratio_notes.tex
```

---

## ETAPA 0 — Coleta de dados: STATUS

### Mercados — coleta de componentes

| Mercado | Status | Fonte | Obs |
|---------|--------|-------|-----|
| S&P 500 | ✅ | Wikipedia, coluna "Symbol" (GitHub CSV fallback) | 503 ativos |
| Ibovespa | ✅ | B3 API oficial + sufixo `.SA` | 79 ativos |
| Nifty 50 | ✅ | Wikipedia, coluna "Symbol" + `.NS` | 50 ativos |
| FTSE 100 | ✅ | Wikipedia, coluna "Ticker" + `.L` (era "EPIC") | 100 ativos |
| DAX 40 | ✅ | Wikipedia, coluna "Ticker" sem sufixo (já tem `.DE`) | 36 válidos* |
| Nikkei 225 | ✅ | Wikipedia API batch (padrão `TYO\|XXXX`) + `.T` | 167 ativos** |
| IPC México | ⬜ pendente | manual | passar `tickers=[...]` |
| CSI 300 | ⬜ pendente | manual | passar `tickers=[...]` |

> *DAX expandiu de 30 para 40 membros em Set/2021. Filtro `present_from_start`
> mantém 36 tickers com cobertura 2015-2024. `ffill(limit=5)` corrige gaps de feriados.
>
> **Nikkei sem tabela estruturada na Wikipedia. Extração via Wikipedia API batch
> lendo `{{TYO|XXXX}}` dos infoboxes das empresas. Re-extrair com `_fetch_nikkei_tickers.py`
> se necessário. IDs 200-222 podem falhar por rate-limiting.

### Controles macro — 29 variáveis diárias, 2015–2026

| Variável | Fonte | Código | Status |
|----------|-------|--------|--------|
| VIX | yfinance | `^VIX` | ✅ |
| DXY | yfinance | `DX-Y.NYB` | ✅ |
| DFF (Fed Funds) | FRED | `DFF` | ✅ |
| SOFR | FRED | `SOFR` | ✅ |
| SONIA | FRED | `IUDSOIA` | ✅ |
| T10Y2Y (curva) | FRED | `T10Y2Y` | ✅ |
| MOVE Index | FRED | indisponível gratuitamente | ❌ |
| GPR + variantes | Caldara & Iacoviello | Excel público | ✅ (mensal) |
| EPU | Baker, Bloom & Davis | Excel público | ✅ (mensal) |
| EMBI (10 países) | IADB / local Excel | `Serie_Historica_Spread_del_EMBI.xlsx` | ✅ |
| CDS soberano | Refinitiv/Markit | manual | ⬜ acesso pago |
| Surpresas FOMC | Fed NY | CSV público | ⬜ pendente |

### Moderadores institucionais — 9 séries anuais, 2000–2024

| Variável | Fonte | Código WB | Cobertura |
|----------|-------|-----------|-----------|
| Rule of Law | World Bank | `GOV_WGI_RL.EST` | 98% |
| Regulatory Quality | World Bank | `GOV_WGI_RQ.EST` | 98% |
| Govt Effectiveness | World Bank | `GOV_WGI_GE.EST` | 98% |
| Control of Corruption | World Bank | `GOV_WGI_CC.EST` | 98% |
| Political Stability | World Bank | `GOV_WGI_PV.EST` | 98% |
| Voice & Accountability | World Bank | `GOV_WGI_VA.EST` | 98% |
| MktCap/GDP (%) | World Bank | `CM.MKT.LCAP.GD.ZS` | 93% |
| Portfolio Inflows (USD) | World Bank | `BX.PEF.TOTL.CD.WD` | 100% |
| GDP (USD) | World Bank | `NY.GDP.MKTP.CD` | 100% |

> **Nota técnica:** WGI não está na base padrão WDI do World Bank. Requer
> endpoint REST `/v2/country/{iso3}/indicator/GOV_WGI_*.EST`. wbgapi falha
> neste endpoint — usar `requests` direto.

---

## ENSAIO 2 — Resultados preliminares (6 mercados)

### CSAD — diagnóstico

| Métrica | Valor |
|---------|-------|
| Período | 2015-01-05 → 2026-05-29 |
| Obs (dias) | 2.867 |
| Ativos/dia (mediana) | 488 |
| NaN médio | 3,2% |
| CSAD médio | 1,112% |
| CSAD máximo | 6,093% (mar/2020) |
| corr(CSAD, \|Rm\|) | 0,597 |

### CCK simétrico — todos os mercados, full vs sub-sample (2019–2024)

| Mercado | β₂ Full | p | β₂ Sub | p | Nota |
|---------|---------|---|--------|---|------|
| S&P 500 | **+0,931** | 0,011** | **+0,720** | 0,092* | Único significativo |
| Ibovespa | +0,355 | 0,357 | +0,385 | 0,306 | ns ambos |
| Nifty 50 | +0,322 | 0,368 | +0,014 | 0,970 | β₂→0 no sub |
| FTSE 100 | +1,365 | 0,403 | +0,544 | 0,678 | ns ambos |
| DAX 40 | +0,782 | 0,174 | **-0,043** | 0,927 | negativo no sub, ns |
| Nikkei 225 | +0,355 | 0,289 | **+0,649** | 0,002*** | anti-herding sub |

> Nenhum mercado mostra β₂ < 0 significativo. DAX sub-amostra é o mais próximo
> (β₂ = -0,043) mas não é significativo. Nifty 50 sub: β₂ ≈ 0 — dispersão
> colapsa para o retorno de mercado nos anos recentes.
>
> Limitação central: CCK testa convergência de magnitudes, não herding direcional.
> O PC1-share captura o herding direcional corretamente (ver abaixo).

### PC1-share (rolling 63 dias) — todos os mercados

| Mercado | PC1 médio | PC1 máx | Excess máx | Pico |
|---------|-----------|---------|------------|------|
| S&P 500 | 0,318 | 0,740 | 25,2× | COVID mar/2020 |
| Ibovespa | 0,353 | 0,783 | 12,5× | COVID mar/2020 |
| Nifty 50 | 0,283 | 0,642 | 8,7× | COVID mar/2020 |
| FTSE 100 | 0,322 | 0,656 | 12,7× | COVID mar/2020 |
| DAX 40 | 0,388 | 0,756 | 8,8× | COVID mar/2020 |
| Nikkei 225 | 0,398 | 0,683 | 16,4× | Out/2024 |

> Eventos marcados no plot: China/Set 2015, Brexit/Jun 2016, Volmageddon/Fev 2018,
> COVID/Mar 2020, Fed hike/Mar 2022, Liberation Day/Abr 2025.
>
> DAX e Nikkei têm PC1-share médio mais alto (0,39-0,40) — menor número de
> componentes cria sincronização estrutural mais forte.
>
> Nikkei pico Out/2024: evento idiossincrático japonês (não global).
> Ibovespa Out/2018 ignorado: eleições brasileiras (fator idiossincrático).

---

## Limitações identificadas no CCK

1. **CCK testa convergência de magnitudes, não herding direcional.**
   Durante um sell-off, todos vendem mas em magnitudes diferentes (airlines
   −20%, tech −5%) → CSAD sobe → β₂ > 0 → CCK lê anti-herding mesmo que
   o comportamento seja de manada.

2. **Quantile regression em Q_τ(CSAD|X) condiciona na variável errada.**
   τ=0,90 captura dias onde CSAD está acima do esperado dado |Rm|, não dias
   onde |Rm| é extremo. O teste correto condiciona nos quantis de |Rm|.

3. **Média ao longo da amostra completa dilui o sinal.**
   Herding é episódico. O β₂ de amostra completa mistura regimes calmos e
   de crise, neutralizando qualquer sinal de herding em sub-períodos.

4. **PC1-share e CSAD divergem nos episódios mais relevantes.**
   COVID: ambos explodem (choque comum, magnitudes heterogêneas).
   Liberation Day: PC1-share explode, CSAD contido → herding genuíno.
   O CCK não consegue distinguir os dois casos.

---

## Nota metodológica — PC1-share como medida de herding

A medida `PC1-share = λ₁ / N` é a proporção da variância cross-sectional
explicada pelo primeiro componente principal (Pearson, 1901; Hotelling, 1933),
calculada em janela móvel de 63 dias.

$$\text{PC1-share}^{(t)} = \frac{\lambda_1^{(t)}}{N}$$

Esta identidade segue de $\sum_j \lambda_j = \text{tr}(\mathbf{C}^{(t)}) = N$
(matriz de correlação: diagonal de 1s).

Captura herding direcional (todos se movem na mesma direção) sem exigir
convergência de magnitudes. Documentação matemática completa em
`absorption_ratio_notes.tex`.

---

## ENSAIO 1 — Revisão sistemática e meta-regressão (Jul–Nov/2026)

### Passos a implementar

1. **`prisma_screen.py`** — automação parcial do funil PRISMA
2. **`bibliometrics.py`** — co-citação, palavras-chave (`networkx`)
3. **`meta_regression.py`** — meta-regressão de efeitos aleatórios (WLS, pesos = 1/SE²)
   - Teste de Egger para viés de publicação
   - Forest plot por tipo de mercado

---

## ENSAIO 2 — Pendências (Dez/2026–Jun/2027)

- [ ] Corrigir scrapers FTSE 100, DAX 40, Nikkei 225
- [ ] Coletar tickers manuais IPC e CSI 300
- [ ] Replicar CCK para todos os 8 mercados
- [ ] `panel_institutional.py` — PanelOLS bidirectional FE (linearmodels)
  - Driscoll-Kraay SE + block bootstrap (block = 22 dias)
  - Moderadores: Regulatory Quality, Cap/PIB, EMBI
- [ ] `stats.py` — block bootstrap para N pequeno

---

## ENSAIO 3 — Regimes e NSS (Jul/2027–Jan/2028)

### Implementar

1. **`realized_vol.py`** — vol realizada 22 dias
2. **`garch.py`** — GARCH(1,1) por mercado (`arch`)
3. **`regime_quantile.py`** — classificação por quantis + Markov-switching
   - Extensão: PC1-share como variável de regime (já disponível)
4. **`nss_pipeline.py`** — Narrative Stress Score via LLM
   - Fontes: NewsAPI, GDELT, Reuters RSS
   - 6 categorias de narrativa: panic, euphoria, bank_crisis, inflation, geopolitics, bubble
   - Modelo: claude-sonnet ou gpt-4o, JSON estruturado, kappa ≥ 0,70
5. **`event_study_covid.py`** — DiD em torno de 23/mar/2020

---

## Controles macro — status de coleta

| Variável | Fonte | Status |
|----------|-------|--------|
| VIX, DXY | yfinance | ✅ |
| DFF, SOFR, SONIA, T10Y2Y | FRED (`fredapi`) | ✅ |
| MOVE Index | FRED / Bloomberg | ❌ indisponível gratuito |
| GPR (6 variantes) | Caldara & Iacoviello | ✅ mensal |
| EPU | Baker, Bloom & Davis | ✅ mensal |
| EMBI (10 países) | Excel local IADB | ✅ diário 2007–2026 |
| CDS soberano | Refinitiv / Markit | ⬜ acesso pago |
| Surpresas FOMC | Fed NY | ⬜ pendente |

---

## Econometria e robustez — checklist

- [ ] Testes de raiz unitária: Im-Pesaran-Shin, Pesaran CIPS
- [ ] Dependência cross-section: Pesaran CD test
- [ ] Autocorrelação: Wooldridge test para painéis
- [ ] Erros-padrão: Driscoll-Kraay + cluster país + block bootstrap
- [ ] Winsorização: retornos 1%–99% por mercado
- [ ] Frequência semanal como robustez
- [ ] Sub-amostras: crise vs. não-crise, pré/pós COVID

---

## Próximos passos (Jun/2026)

1. ✅ `fetch_market.py` — coleta com cache, 6 mercados funcionando
2. ✅ `fetch_controls.py` — 29 controles macro
3. ✅ `fetch_institutional.py` — WGI + moderadores
4. ✅ `csad.py` — CSAD + diagnóstico
5. ✅ `cck_symmetric.py` — CCK OLS + assimétrico
6. ✅ `cck_quantile.py` — CCK quantile
7. ✅ `absorption_ratio.py` — PC1-share rolling
8. ✅ Corrigir FTSE 100, DAX 40 scrapers + Nikkei 225 via Wikipedia API
9. ✅ `03_all_markets.py` — pipeline completo 6 mercados + CCK + PC1-share
10. ✅ Apresentação Beamer 14 slides com resultados multi-mercado
11. ⬜ Adicionar IPC (México) e CSI 300 (China) — tickers manuais
12. ⬜ `panel_institutional.py` — PanelOLS 8 países com moderadores
13. ⬜ CCK assimétrico para todos os mercados
14. ⬜ Definir orientação formal e pré-registrar desenho metodológico
15. ⬜ Solicitar acesso Refinitiv/Bloomberg (MOVE, CDS)
16. ⬜ Iniciar busca PRISMA (Web of Science export) para Ensaio 1
17. ⬜ Criar repositório Git

---

## Dependências Python (requirements.txt)

```
yfinance>=0.2.40
pandas>=2.1
numpy>=1.26
requests>=2.31
beautifulsoup4>=4.12
fredapi>=0.5
statsmodels>=0.14
linearmodels>=6.0
arch>=6.3
scikit-learn>=1.4
matplotlib>=3.8
openpyxl>=3.1
anthropic>=0.25
openai>=1.30
tqdm>=4.66
pyarrow>=15.0
jupyter>=1.0
```
