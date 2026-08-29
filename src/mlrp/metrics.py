"""Mesures de prédiction (identiques à 2024) et de performance (TCAC en années civiles)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

TRADING_DAYS = 252


# ----------------------------------------------------------------------------- prédiction (code 2024)
def r_squared_modified(y_true, y_pred, y_train) -> float:
    """R² hors échantillon avec, au dénominateur, la moyenne de l'échantillon d'entraînement (code 2024).

    Les valeurs manquantes sont écartées explicitement. Le code de 2024 employait ``np.mean`` sur
    ``y_train`` : un seul rendement manquant, celui d'un titre entré en bourse après le début de
    l'échantillon, suffisait à rendre la moyenne NaN, donc le R² NaN, sans message. C'est ce qui
    vidait la colonne ``r2_oos_pooling`` du panel américain (mesuré au 2026-08-29).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    garde = np.isfinite(y_true) & np.isfinite(y_pred)
    if garde.sum() == 0:
        return float("nan")
    moyenne = np.nanmean(np.asarray(y_train, dtype=float))
    ssr = np.sum((y_true[garde] - y_pred[garde]) ** 2)
    sst = np.sum((y_true[garde] - moyenne) ** 2)
    return 1 - ssr / sst


def _pt_inputs(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.dtype.kind in "fc" and np.any((y_pred > 0) & (y_pred < 1)):
        y_pred = np.where(y_pred >= 0.5, 1, 0)
    else:
        y_pred = y_pred.astype(int)
    y_true = np.where(y_true >= 0.5, 1, 0)
    return y_true, y_pred


def pesaran_timmermann_stat(y_true, y_pred) -> float:
    """Statistique de Pesaran-Timmermann (1992) de justesse directionnelle, telle que codée en 2024."""
    y_true, y_pred = _pt_inputs(y_true, y_pred)
    n = y_true.shape[0]
    dac = np.sum(y_true == y_pred) / n
    p_true = np.sum(y_true == 1) / n
    var_true = p_true * (1 - p_true) / n
    p_pred = np.sum(y_pred == 1) / n
    var_pred = p_pred * (1 - p_pred) / n
    expected = p_true * p_pred + (1 - p_true) * (1 - p_pred)
    var_expected = expected * (1 - expected) / n
    correction = ((2 * p_true - 1) ** 2) * var_pred + ((2 * p_pred - 1) ** 2) * var_true + 4 * var_true * var_pred
    diff = var_expected - correction
    return 0.0 if diff <= 0 else float((dac - expected) / np.sqrt(diff))


def pesaran_timmermann_p_value(y_true, y_pred) -> float:
    return float(1 - stats.norm.cdf(pesaran_timmermann_stat(y_true, y_pred)))


METRICS = {
    "mean_squared_error": mean_squared_error,
    "accuracy_score": accuracy_score,
    "f1_score": f1_score,
    "r_squared_modified": r_squared_modified,
    "pesaran_timmermann_stat": pesaran_timmermann_stat,
    "pesaran_timmermann_p_value": pesaran_timmermann_p_value,
}

FAMILY_METRICS = {
    "regressor": (["r_squared_modified", "mean_squared_error"], "maximize"),
    "classifier": (["pesaran_timmermann_p_value", "pesaran_timmermann_stat", "accuracy_score", "f1_score"], "minimize"),
}


# ------------------------------------------------------------------------------------- performance
def cagr(returns: pd.Series) -> float:
    """Taux de croissance annuel composé en années civiles : (1 + R)^(365,25 / jours) - 1."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    years = (r.index[-1] - r.index[0]).days / 365.25
    total = float((1 + r).prod())
    return total ** (1 / years) - 1 if years > 0 and total > 0 else float("nan")


def annual_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std() * np.sqrt(periods))


_EPS = 1e-12


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf / periods
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd > _EPS else float("nan")


def sortino(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf / periods
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > _EPS else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    r = returns.dropna() - threshold
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else float("inf")


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns.dropna()).prod() - 1)


def performance_table(returns: pd.Series, name: str) -> pd.Series:
    return pd.Series(
        {"Cumulative_Returns": cumulative_return(returns), "CAGR": cagr(returns), "Sharpe": sharpe(returns),
         "Volatility": annual_volatility(returns), "Max_Drawdown": max_drawdown(returns),
         "Sortino": sortino(returns), "Omega_Ratio": omega(returns)},
        name=name,
    )


def r2_oos_by_level(y_true: pd.DataFrame, y_pred: pd.DataFrame, y_train: pd.DataFrame) -> pd.Series:
    """R² hors échantillon par titre, plus la moyenne et la version « pooling » (toutes séries empilées)."""
    out = {}
    for col in y_pred.columns:
        out[col] = r_squared_modified(y_true[col].loc[y_pred.index], y_pred[col], y_train[col])
    s = pd.Series(out)
    yt = y_true.loc[y_pred.index, y_pred.columns].values.ravel()
    yp = y_pred.values.ravel()
    ok = ~(np.isnan(yt) | np.isnan(yp))  # sans ce masque, un seul NaN rendait le « pooling » NaN
    pooled = r_squared_modified(yt[ok], yp[ok], y_train[y_pred.columns].values.ravel())
    s.loc["average"] = float(s.mean())
    s.loc["pooling"] = float(pooled)
    return s
