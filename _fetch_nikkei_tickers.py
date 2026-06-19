"""Helper script — extrai tickers do Nikkei 225 via Wikipedia API."""
import re
import sys
import time
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {"User-Agent": "research/1.0"}

# --- 1. Pega links do navbox ---
r = requests.get("https://en.wikipedia.org/wiki/Nikkei_225",
                 headers=HEADERS, timeout=15)
soup = BeautifulSoup(r.text, "html.parser")

pages = []
for nav in soup.find_all("div", class_="navbox"):
    if "nikkei" in nav.get_text()[:120].lower():
        for a in nav.find_all("a", href=True):
            href = a.get("href", "")
            if href.startswith("/wiki/") and ":" not in href:
                title = unquote(href.replace("/wiki/", "")).replace("_", " ")
                pages.append(title)
        break

pages = list(dict.fromkeys(pages))
pages = [p for p in pages if p not in ("Japan",)]
print(f"Pages collected: {len(pages)}")

# --- 2. Wikipedia API em batches de 20 ---
API = "https://en.wikipedia.org/w/api.php"
tickers = {}

for i in range(0, len(pages), 20):
    batch = pages[i : i + 20]
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": "|".join(batch),
    }
    try:
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"  [erro batch {i}]: {e}")
        time.sleep(1)
        continue

    for pg in data.get("query", {}).get("pages", {}).values():
        title = pg.get("title", "")
        revs  = pg.get("revisions", [])
        if not revs:
            continue
        content = revs[0].get("slots", {}).get("main", {}).get("*", "")
        for pat in [r"TYO\|(\d{4,5})", r"TSE\|(\d{4,5})",
                    r"traded_as\s*=.*?(\d{4,5})\s*\}\}"]:
            m = re.search(pat, content)
            if m:
                tickers[title] = m.group(1) + ".T"
                break

    print(f"  batch {i}–{i+20}: {len(tickers)} tickers so far")
    time.sleep(0.3)

print(f"\nTotal tickers found: {len(tickers)} / {len(pages)}")
print("\nSample:")
for k, v in list(tickers.items())[:15]:
    print(f"  {k}: {v}")

# Salva resultado
import json
out = sorted(tickers.values())
with open("_nikkei225_tickers.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSalvo: _nikkei225_tickers.json ({len(out)} tickers)")
