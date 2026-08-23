"""Compare, pour une feuille de résultats du mémoire, le portefeuille « tel que publié » et le vrai long-short.

Entrées : les poids long/short et les rendements de stratégie archivés (results/archive_2024/...), plus les prix
quotidiens bruts (data/raw_data/<pays>.csv). Sorties : un tableau CSV dans results/tables/ et un résumé à l'écran.

Usage :
    python scripts/compare_long_short_modes.py --country usa --strategy ridge_regressor --period "2008-01 à 2024-01" --signal "Top 10"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ml_returns_pred.compute_strategy_returns.strategy_returns_calculator import StrategyReturnsCalculator  # noqa: E402
from ml_returns_pred.paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor  # noqa: E402

PRICES = {
    "usa": "us_stocks_2000-01-01_to_2024-06-01.csv",
    "canada": "canadian_stocks_2000-01-01_to_2024-06-01.csv",
}
COUNTRY_DIR = {"usa": "US Final", "canada": "Canada Final"}


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float((1 + r).prod() ** (1 / years) - 1)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(r: pd.Series) -> float:
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min())


def run(country: str, strategy: str, period: str, signal: str, max_date: str) -> pd.DataFrame:
    sheet = RESULTS_DIR / "archive_2024" / {"usa": "usa", "canada": "canada"}[country] / period / signal

    def _read(name: str) -> pd.DataFrame:
        for ext in (".csv.gz", ".csv"):
            p = sheet / f"{name}{ext}"
            if p.exists():
                return pd.read_csv(p, index_col=0, parse_dates=True)
        raise FileNotFoundError(f"{name} introuvable dans {sheet}")

    lw = _read(f"{strategy}_long_weights")
    sw = _read(f"{strategy}_short_weights")
    archived = _read(f"{strategy}_strategy_returns").iloc[:, 0]

    prices = pd.read_csv(DATA_DIR / "raw_data" / PRICES[country], index_col=0, parse_dates=True)
    dp = DataPreprocessor()
    prices = dp.keep_data_until_max_date(data=dp.preprocess(data=prices), max_date=max_date)

    rows = {}
    for mode in ("as_published", "corrected"):
        calc = StrategyReturnsCalculator(long_weights=lw, short_weights=sw, prices_data_preprocessed=prices,
                                         transaction_fee=0.0, is_long_only=False, long_short_mode=mode)
        calc.calculate_drifted_weights()
        r = calc.compute_strategy_returns()["Portfolio_Returns"]
        dl, ds = calc.drifted_weights_long.astype(float), calc.drifted_weights_short.astype(float)
        dr = calc.daily_returns.loc[dl.index[0]:]
        long_leg = (dl.shift(1) * dr).sum(axis=1)
        short_leg = (ds.shift(1) * dr).sum(axis=1)
        rows[mode] = {
            "CAGR": cagr(r), "Sharpe": sharpe(r), "Volatilite": float(r.std() * np.sqrt(252)), "MaxDD": max_dd(r),
            "CAGR_jambe_longue": cagr(long_leg), "CAGR_jambe_courte_tenue_longue": cagr(short_leg),
            "exposition_brute": float((dl.abs().sum(axis=1) + ds.abs().sum(axis=1)).median()),
            "ecart_max_vs_archive": float((r - archived).abs().max()) if mode == "as_published" else np.nan,
        }
    table = pd.DataFrame(rows).T
    out = RESULTS_DIR / "tables" / f"long_short_modes_{country}_{strategy}_{signal.replace(' ', '').lower()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(table.round(4).to_string())
    print(f"\nécrit : {out}")
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", choices=list(PRICES), default="usa")
    ap.add_argument("--strategy", default="ridge_regressor")
    ap.add_argument("--period", default="2008-01 à 2024-01")
    ap.add_argument("--signal", default="Top 10")
    ap.add_argument("--max-date", default="2024-01-01")
    a = ap.parse_args()
    run(a.country, a.strategy, a.period, a.signal, a.max_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
