"""Des prédictions aux rendements de portefeuille : rangs, poids, dérive vectorisée, coûts.

Deux modes : ``corrected`` (long - short, dérive (1 + r) des deux jambes, coûts sur la rotation réelle) et
``as_published`` (code 2024 : jambes additionnées, dérive (1 - r) des deux jambes ; les poids du mémoire sont
reproduits à l'identique, voir tests/test_v2_portfolio.py). La dérive quotidienne du code de 2024 renormalise
chaque jambe à 100 % tous les jours ; cette convention est conservée dans les deux modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------- signaux
def rank_predictions(y_pred: pd.DataFrame) -> pd.DataFrame:
    """Rang 1 = prédiction la plus élevée, méthode « first » (ordre d'apparition pour les ex aequo)."""
    return y_pred.dropna(axis=0, how="all").rank(axis=1, ascending=False, method="first")


def top_k_masks(ranks: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jambe longue : rangs <= k ; jambe courte : rangs >= (rang maximal global - k + 1), comme en 2024."""
    long_mask = ranks <= k
    short_mask = ranks >= (ranks.max().max() - k + 1)
    return long_mask, short_mask


def sign_masks(y_pred: pd.DataFrame, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal « positif » : long si la prévision est > 0 (régression) ou la classe 1 (classification), court sinon."""
    y = y_pred.dropna(axis=0, how="all")
    long_mask = y > 0 if family == "regressor" else y >= 0.5
    return long_mask, ~long_mask & y.notna()


def equal_weights(mask: pd.DataFrame) -> pd.DataFrame:
    w = mask.astype(float)
    n = w.sum(axis=1)
    return w.div(n.where(n > 0, np.nan), axis=0).fillna(0.0)


def build_weights(y_pred: pd.DataFrame, signal: str, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal in ("top10", "top20"):
        k = 10 if signal == "top10" else 20
        long_mask, short_mask = top_k_masks(rank_predictions(y_pred), k)
    elif signal == "positive":
        long_mask, short_mask = sign_masks(y_pred, family)
    else:
        raise ValueError(signal)
    return equal_weights(long_mask), equal_weights(short_mask)


# --------------------------------------------------------------------------------------------- dérive
def _drift_block(w0: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Poids dérivés sur un bloc de jours (ligne 0 = jour de rééquilibrage, poids cibles).

    w_t = normalise(w_0 * prod_{s<=t} f_s) avec f_s = 0 dès qu'un rendement manque (le titre sort) ; la
    normalisation (somme des valeurs absolues = 1) n'est appliquée que si plus d'un poids est non nul.
    """
    cum = np.cumprod(np.vstack([np.ones_like(factors[:1]), factors[1:]]), axis=0)
    w = w0 * cum
    nonzero = (w != 0).sum(axis=1)
    norm = np.abs(w).sum(axis=1)
    scale = np.where((nonzero > 1) & (norm > 0), norm, 1.0)
    return w / scale[:, None]


def rebalance_positions(target_index: pd.DatetimeIndex, daily_index: pd.DatetimeIndex,
                        alignment: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions (dans l'index quotidien) où chaque ligne de poids cibles est appliquée.

    ``"next_trading_day"`` : premier jour de bourse à partir de la date cible (comportement attendu).
    ``"legacy"`` : code de 2024 : la première ligne s'applique au premier jour disponible, mais les dates
    suivantes ne s'appliquent que si elles tombent exactement sur un jour de bourse ; un premier du mois
    tombant un samedi, un dimanche ou un jour férié est donc un rééquilibrage SAUTÉ (les poids continuent
    de dériver). Retourne (positions, indices des lignes cibles retenues).
    """
    if alignment == "next_trading_day":
        pos = daily_index.searchsorted(target_index, side="left")
        keep = pos < len(daily_index)
        pos, rows = pos[keep], np.arange(len(target_index))[keep]
        # deux cibles sur le même jour : la dernière l'emporte
        _, last = np.unique(pos[::-1], return_index=True)
        sel = np.sort(len(pos) - 1 - last)
        return pos[sel], rows[sel]
    if alignment == "legacy":
        pos = daily_index.get_indexer(target_index)
        rows = np.arange(len(target_index))
        if len(pos) and pos[0] < 0:
            first = daily_index.searchsorted(target_index[0], side="left")
            pos[0] = first if first < len(daily_index) else -1
        keep = pos >= 0
        return pos[keep], rows[keep]
    raise ValueError(alignment)


def drifted_weights(target: pd.DataFrame, daily_returns: pd.DataFrame, factor_sign: float,
                    alignment: str = "next_trading_day") -> pd.DataFrame:
    """Poids quotidiens dérivés, vectorisés par bloc de rééquilibrage.

    ``factor_sign`` = +1 pour (1 + r), -1 pour (1 - r) (convention du code de 2024 en long-short) ;
    ``alignment`` : voir ``rebalance_positions``.
    """
    target, r = target.align(daily_returns, axis=1, join="inner")
    target = target.fillna(0.0)
    start = max(target.index.min(), r.index.min())
    r = r.loc[start:]
    target = target.loc[start:]
    factors = (1 + factor_sign * r.values).astype(float)
    factors[np.isnan(r.values)] = 0.0

    out = np.zeros(r.shape)
    rebal_pos, rows = rebalance_positions(target.index, r.index, alignment)
    bounds = list(rebal_pos) + [len(r)]
    for i, start_pos in enumerate(rebal_pos):
        end_pos = bounds[i + 1]
        w0 = target.iloc[rows[i]].values.astype(float)
        out[start_pos:end_pos] = _drift_block(w0, factors[start_pos:end_pos])
    return pd.DataFrame(out, index=r.index, columns=r.columns)


@dataclass
class StrategyResult:
    returns: pd.Series            # rendements quotidiens nets
    long_weights: pd.DataFrame    # poids dérivés de la jambe longue
    short_weights: pd.DataFrame | None
    turnover: pd.Series           # rotation (somme des |variations|) aux dates de rééquilibrage
    gross_exposure: pd.Series     # somme des |poids| des deux jambes


def strategy_returns(long_target: pd.DataFrame, short_target: pd.DataFrame | None, prices_daily: pd.DataFrame,
                     mode: str = "corrected", fee: float = 0.0) -> StrategyResult:
    """Rendements quotidiens d'un portefeuille rééquilibré aux dates des poids cibles.

    Les prix manquants sont reportés (``ffill``) avant le calcul des rendements : convention du code de 2024
    (un jour sans cotation donne un rendement nul, un titre radié reste à plat dans le portefeuille au lieu
    d'en sortir). Rendu explicite pour survivre aux changements de défaut de ``pct_change`` dans pandas 3.
    """
    daily = prices_daily.ffill().pct_change(fill_method=None).iloc[1:]
    long_only = short_target is None
    sign = 1.0 if (mode == "corrected" or long_only) else -1.0
    alignment = "legacy" if mode == "as_published" else "next_trading_day"
    wl = drifted_weights(long_target, daily, sign, alignment)
    r = daily.loc[wl.index[0]:, wl.columns]
    ret = (wl.shift(1) * r).sum(axis=1)
    ws = None
    if not long_only:
        ws = drifted_weights(short_target, daily, sign, alignment)
        short_leg = (ws.shift(1) * r).sum(axis=1)
        ret = ret - short_leg if mode == "corrected" else ret + short_leg

    # rotation aux jours de rééquilibrage effectifs : poids appliqués contre poids dérivés de la veille.
    # L'index complet des cibles est utilisé : une cible antérieure au premier jour de bourse (1er tombant un
    # week-end) est rattachée au premier jour, comme dans la dérive ; sa rotation de mise en place est forcée à 0.
    pos, _ = rebalance_positions(long_target.index, wl.index, alignment)
    rebal = wl.index[pos]
    turn = (wl.loc[rebal] - wl.shift(1).loc[rebal].fillna(0.0)).abs().sum(axis=1)
    if ws is not None:
        turn = turn + (ws.loc[rebal] - ws.shift(1).loc[rebal].fillna(0.0)).abs().sum(axis=1)
    if len(turn):
        turn.iloc[0] = 0.0
    costs = pd.Series(0.0, index=ret.index)
    costs.loc[turn.index] = fee * turn.values
    net = (ret - costs).rename("Portfolio_Returns")
    gross = wl.abs().sum(axis=1) + (ws.abs().sum(axis=1) if ws is not None else 0.0)
    return StrategyResult(returns=net, long_weights=wl, short_weights=ws, turnover=turn, gross_exposure=gross)


def equally_weighted_long_only(returns_monthly: pd.DataFrame, prices_daily: pd.DataFrame, start: str | None = None,
                               fee: float = 0.0) -> StrategyResult:
    """Portefeuille de référence : tous les titres disponibles, poids égaux, rééquilibré chaque mois."""
    mask = returns_monthly.notna()
    if start is not None:
        mask = mask.loc[pd.Timestamp(start):]
    target = equal_weights(mask)
    return strategy_returns(target, None, prices_daily, mode="corrected", fee=fee)
