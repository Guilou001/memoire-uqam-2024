"""Télécharge les données brutes dans data/raw_data/ (idempotent).

- Prix quotidiens des 49 titres canadiens et des 50 titres américains, et des indices de référence
  (^GSPTSE -> S&P/TSX composite, ^GSPC -> S&P 500, ^IXIC -> NASDAQ) via yfinance, 2000-01-01 -> 2024-06-01, prix de
  clôture ajustés (auto_adjust=True), comme dans les fichiers utilisés par le mémoire.
- FRED-MD (McCracken et Ng) : à déposer à la main sous ``data/raw_data/Fred-MD.csv`` (séparateur « ; »,
  colonne ``sasdate``) ; le script indique l'URL. Le mémoire a utilisé le millésime de juin 2024.
- LCDMA (Fortin-Gagnon, Leroux, Stevanovic, Surprenant, 2022) : à déposer sous ``data/raw_data/macro_data.csv``
  (panel mensuel équilibré, colonne ``Date``, 410 variables) ; le mémoire a utilisé le millésime de juillet 2024
  (``balanced_can_md.csv``), disponible sur https://www.stevanovic.uqam.ca/DS_LCMD.html.

Attention : les prix ajustés de Yahoo sont révisés dans le temps (dividendes, corrections). Un
téléchargement en 2026 ne redonne donc pas à l'identique les prix canadiens de 2024 ; voir README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ml_returns_pred.cli import CANADIAN_TICKERS, US_TICKERS  # noqa: E402
from ml_returns_pred.paths import DATA_DIR  # noqa: E402

RAW = DATA_DIR / "raw_data"
START, END = "2000-01-01", "2024-06-01"
# la clé « TSX60 » est le nom de FICHIER hérité de 2024 ; le ticker est celui du composite
BENCHMARKS = {"TSX60": "^GSPTSE", "SP500": "^GSPC", "NASDAQ": "^IXIC"}


def download_prices(tickers: list[str], name: str) -> Path:
    import yfinance as yf

    out = RAW / f"{name}_{START}_to_{END}.csv"
    if out.exists():
        print(f"déjà présent : {out.name}")
        return out
    data = yf.download(tickers=tickers, start=START, end=END, interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        frame = pd.DataFrame({t: data[t]["Close"] for t in tickers})
    else:  # un seul ticker
        frame = data[["Close"]].rename(columns={"Close": name})
    frame.index = pd.to_datetime(frame.index).normalize()
    frame.index.name = "Date"
    frame.to_csv(out)
    print(f"écrit : {out.name} {frame.shape}")
    return out


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    download_prices(CANADIAN_TICKERS, "canadian_stocks")
    download_prices(US_TICKERS, "us_stocks")
    for name, ticker in BENCHMARKS.items():
        download_prices([ticker], name)
    for fname, url in [
        ("Fred-MD.csv", "https://www.stlouisfed.org/research/economists/mccracken/fred-databases (millésime mensuel, colonne sasdate, séparateur ;)"),
        ("macro_data.csv", "https://www.stevanovic.uqam.ca/DS_LCMD.html (balanced_can_md.csv du zip LCDMA, colonne Date)"),
    ]:
        if not (RAW / fname).exists():
            print(f"MANQUANT : {fname} -> à télécharger manuellement depuis {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
