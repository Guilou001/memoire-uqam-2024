"""Chargement et préparation des données, équivalents vectorisés du code de 2024.

Étapes (même ordre que le mémoire) : lecture des prix quotidiens, des variables macro mensuelles et de
l'indice ; remplissage avant des prix entre première et dernière valeur connue ; suppression des colonnes
macro incomplètes ; alignement sur la période commune ; troncature à la date maximale ; rééchantillonnage
des prix sur les dates macro (début de mois) ; rendements arithmétiques ; binarisation pour les classifieurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlrp.config import COUNTRIES, RAW_DIR


# --------------------------------------------------------------------------------------------- lecture
def read_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce", format="%Y-%m-%d")
    return df.apply(pd.to_numeric, errors="coerce")


def read_macro(path: Path, kind: str) -> pd.DataFrame:
    """FRED-MD : séparateur « ; » et colonne ``sasdate`` ; LCDMA : virgule et colonne ``Date``."""
    if kind == "fredmd":
        df = pd.read_csv(path, index_col=0, delimiter=";", parse_dates=["sasdate"])
    else:
        df = pd.read_csv(path, index_col=0, delimiter=",", parse_dates=["Date"])
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df.apply(pd.to_numeric, errors="coerce")


def read_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, sep=",", parse_dates=["Date"])
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    return df.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------------------- prétraitement
def forward_fill_within_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Remplit vers l'avant chaque colonne entre sa première et sa dernière valeur non manquante seulement."""
    notna = prices.notna()
    inside = notna.cummax() & notna.iloc[::-1].cummax().iloc[::-1]
    return prices.ffill().where(inside)


def drop_incomplete_columns(macro: pd.DataFrame) -> pd.DataFrame:
    return macro.dropna(axis=1)


def align_common_period(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = max(a.index.min(), b.index.min())
    end = min(a.index.max(), b.index.max())
    return a.loc[start:end], b.loc[start:end]


def truncate(df: pd.DataFrame, max_date: str) -> pd.DataFrame:
    return df.loc[df.index <= pd.Timestamp(max_date)]


def resample_to_reference(prices: pd.DataFrame, reference: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Réindexe les prix sur les dates de la référence (remplissage avant sur l'union des dates), puis ``asfreq``."""
    common = prices.index.union(reference.index)
    resampled = prices.reindex(common).ffill().reindex(reference.index).dropna(axis=0, how="all")
    return resampled.asfreq(freq=freq, method="ffill")


def monthly_frequency(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    return df.asfreq(freq=freq, method="ffill")


def arithmetic_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().iloc[1:]


def binarize(returns: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    return (returns > threshold).astype(int)


# -------------------------------------------------------------------------------------------- dataset
@dataclass
class Dataset:
    """Données prêtes pour un (pays, période)."""

    country: str
    max_date: str
    prices_daily: pd.DataFrame      # prix quotidiens prétraités et tronqués (pour les rendements de stratégie)
    prices_monthly: pd.DataFrame    # prix rééchantillonnés sur les dates macro
    macro_monthly: pd.DataFrame     # variables macro mensuelles (colonnes complètes)
    returns_monthly: pd.DataFrame   # rendements arithmétiques mensuels (cible des régressions)
    benchmark_daily: pd.DataFrame   # niveaux quotidiens de l'indice

    @property
    def exog_monthly(self) -> pd.DataFrame:
        """Exogènes alignées sur les rendements (on retire la première ligne comme dans le mémoire)."""
        return self.macro_monthly.iloc[1:]

    def fingerprint(self) -> str:
        import hashlib

        h = hashlib.sha1()
        for df in (self.returns_monthly, self.macro_monthly):
            h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        return h.hexdigest()[:12]


def build_dataset(country: str, max_date: str, raw_dir: Path = RAW_DIR) -> Dataset:
    spec = COUNTRIES[country]
    prices = forward_fill_within_history(read_prices(raw_dir / spec["prices"]))
    macro = drop_incomplete_columns(read_macro(raw_dir / spec["macro"], spec["macro_kind"]))
    benchmark = read_benchmark(raw_dir / spec["benchmark"])

    prices_aligned, macro_aligned = align_common_period(prices, macro)
    prices_aligned = truncate(prices_aligned, max_date)
    macro_aligned = truncate(macro_aligned, max_date)

    prices_monthly = resample_to_reference(prices_aligned, macro_aligned)
    macro_monthly = monthly_frequency(macro_aligned)
    returns_monthly = arithmetic_returns(prices_monthly)

    return Dataset(country=country, max_date=max_date, prices_daily=prices_aligned, prices_monthly=prices_monthly,
                   macro_monthly=macro_monthly, returns_monthly=returns_monthly, benchmark_daily=benchmark)


def synthetic_dataset(n_months: int = 60, n_assets: int = 6, n_macro: int = 4, seed: int = 0,
                      start: str = "2010-01-01") -> Dataset:
    """Jeu synthétique (tests et démonstrations) avec la même structure que les données réelles."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_months * 21)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (len(days), n_assets)), axis=0)),
                          index=days, columns=[f"A{i:02d}" for i in range(n_assets)])
    months = pd.date_range(start, periods=n_months, freq="MS")
    macro = pd.DataFrame(rng.normal(size=(n_months, n_macro)), index=months, columns=[f"M{i}" for i in range(n_macro)])
    macro.index.name = "Date"
    bench = pd.DataFrame({"BENCH": 1000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days))))}, index=days)
    prices_aligned, macro_aligned = align_common_period(forward_fill_within_history(prices), macro)
    prices_monthly = resample_to_reference(prices_aligned, macro_aligned)
    macro_monthly = monthly_frequency(macro_aligned)
    return Dataset(country="synthetic", max_date=str(days[-1].date()), prices_daily=prices_aligned,
                   prices_monthly=prices_monthly, macro_monthly=macro_monthly,
                   returns_monthly=arithmetic_returns(prices_monthly), benchmark_daily=bench)
