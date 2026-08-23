"""Le portefeuille vectorisé de mlrp reproduit le code de 2024 (poids, dérive, rendements) dans les deux modes."""

import numpy as np
import pandas as pd
import pytest

from ml_returns_pred.compute_strategy_returns.strategy_returns_calculator import StrategyReturnsCalculator
from ml_returns_pred.create_long_short_portfolio.long_short_portfolio_creator import (
    LongShortPortfolioCreatorFromRanking,
)
from ml_returns_pred.weighting.weighting import WeightingStrategyFactory
from mlrp.portfolio import build_weights, drifted_weights, equally_weighted_long_only, strategy_returns


@pytest.fixture()
def market():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2015-01-01", periods=420)
    n = 12
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), n)), axis=0)),
                          index=days, columns=[f"T{i:02d}" for i in range(n)])
    prices.iloc[:40, 3] = np.nan                     # un titre qui démarre tard
    prices.iloc[300:, 5] = np.nan                    # un titre qui disparaît
    months = pd.date_range("2015-03-01", periods=12, freq="MS")
    y_pred = pd.DataFrame(rng.normal(size=(len(months), n)), index=months, columns=prices.columns)
    return prices, y_pred


def _legacy_weights(y_pred, k):
    creator = LongShortPortfolioCreatorFromRanking(signals=y_pred)
    long_sig, short_sig = creator.create_long_short_portfolio(
        ranking_strategy="simple", ascending=False, method="first", selection_method="fix",
        percentile_threshold=0.1, fix_threshold=k, keep_signal_value=False, use_ranking_as_signals=True)
    strat = WeightingStrategyFactory.select_weighting_strategy(strategy_type="rank_based_weighting", method="equal")
    return strat.compute_weights(long_signals=long_sig, short_signals=short_sig, fix_threshold=k)


@pytest.mark.parametrize("k", [3, 5])
def test_weights_match_legacy(market, k):
    _, y_pred = market
    from mlrp.portfolio import equal_weights, rank_predictions, top_k_masks

    lm, sm = top_k_masks(rank_predictions(y_pred), k)
    lw, sw = equal_weights(lm), equal_weights(sm)
    lw_old, sw_old = _legacy_weights(y_pred, k)
    pd.testing.assert_frame_equal(lw, lw_old.astype(float), check_names=False)
    pd.testing.assert_frame_equal(sw, sw_old.astype(float), check_names=False)


@pytest.mark.parametrize("mode", ["as_published", "corrected"])
def test_strategy_returns_match_legacy(market, mode):
    """as_published : dates cibles au 1er du mois (certaines tombent un week-end, rééquilibrage sauté comme en 2024) ;
    corrected : dates cibles en jours de bourse (même convention des deux côtés), comparaison exacte aussi."""
    prices, y_pred = market
    if mode == "corrected":
        y_pred = y_pred.copy()
        y_pred.index = pd.bdate_range("2015-03-02", periods=len(y_pred), freq="BMS")
    lw, sw = _legacy_weights(y_pred, 3)
    legacy = StrategyReturnsCalculator(long_weights=lw, short_weights=sw, prices_data_preprocessed=prices.ffill(),
                                       transaction_fee=0.0, is_long_only=False, long_short_mode=mode)
    legacy.calculate_drifted_weights()
    old = legacy.compute_strategy_returns()["Portfolio_Returns"]
    new = strategy_returns(lw, sw, prices.ffill(), mode=mode, fee=0.0)
    joined = pd.concat([old, new.returns], axis=1, join="inner")
    assert len(joined) > 200
    assert np.nanmax(np.abs(joined.iloc[:, 0].values - joined.iloc[:, 1].values)) < 1e-12
    pd.testing.assert_frame_equal(new.long_weights.loc[legacy.drifted_weights_long.index],
                                  legacy.drifted_weights_long.astype(float), check_names=False, atol=1e-12)


def test_corrected_mode_never_skips_a_rebalance(market):
    """Quand le 1er du mois tombe un week-end, le code de 2024 saute le rééquilibrage ; la v2 corrigée prend le
    jour de bourse suivant."""
    prices, y_pred = market
    lw, sw = build_weights(y_pred, "top10", "regressor")
    corrected = strategy_returns(lw, sw, prices, mode="corrected")
    published = strategy_returns(lw, sw, prices, mode="as_published")
    weekends = sum(1 for d in y_pred.index[1:] if d not in prices.index)
    assert weekends > 0
    assert len(corrected.turnover) == len(y_pred)
    assert len(published.turnover) == len(y_pred) - weekends


def test_modes_differ_and_exposure(market):
    prices, y_pred = market
    lw, sw = build_weights(y_pred, "top10", "regressor")
    a = strategy_returns(lw, sw, prices, mode="as_published")
    c = strategy_returns(lw, sw, prices, mode="corrected")
    assert not np.allclose(a.returns, c.returns)
    assert abs(a.gross_exposure.median() - 2.0) < 1e-9


def test_positive_signal_and_long_only(market):
    prices, y_pred = market
    lw, sw = build_weights(y_pred, "positive", "regressor")
    assert ((lw > 0) & (sw > 0)).sum().sum() == 0          # jambes disjointes
    np.testing.assert_allclose(lw.sum(axis=1), 1.0)
    ew = equally_weighted_long_only(y_pred.notna().astype(float), prices)
    assert abs(ew.gross_exposure.median() - 1.0) < 1e-9
    assert ew.turnover.iloc[0] == 0.0


def test_costs_reduce_returns(market):
    prices, y_pred = market
    lw, sw = build_weights(y_pred, "top10", "regressor")
    free = strategy_returns(lw, sw, prices, mode="corrected", fee=0.0)
    paid = strategy_returns(lw, sw, prices, mode="corrected", fee=0.001)
    assert paid.returns.sum() < free.returns.sum()
    assert (paid.turnover.iloc[1:] > 0).all()


def test_drift_single_position_not_renormalised():
    days = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41], "B": [100, 100, 100, 100, 100]}, index=days)
    target = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[days[1]])
    w = drifted_weights(target, prices.pct_change().iloc[1:], +1.0)
    assert w.loc[days[3], "A"] == pytest.approx(1.1 * 1.1)  # un seul poids non nul : pas de renormalisation (règle 2024)
