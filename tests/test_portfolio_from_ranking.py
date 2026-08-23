"""Sélection top-k / bottom-k par rang et pondération égale (configuration « top 10 » du mémoire)."""

import numpy as np
import pandas as pd

from ml_returns_pred.create_long_short_portfolio.long_short_portfolio_creator import (
    LongShortPortfolioCreatorFromRanking,
)
from ml_returns_pred.weighting.weighting import WeightingStrategyFactory


def _signals(n_dates=3, n_assets=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="MS")
    cols = [f"T{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(rng.normal(size=(n_dates, n_assets)), index=idx, columns=cols)


def test_top_k_long_and_bottom_k_short_equal_weights():
    sig = _signals()
    creator = LongShortPortfolioCreatorFromRanking(signals=sig)
    long_sig, short_sig = creator.create_long_short_portfolio(
        ranking_strategy="simple", ascending=False, method="first", selection_method="fix",
        percentile_threshold=0.1, fix_threshold=3, keep_signal_value=False, use_ranking_as_signals=True,
    )
    # avec use_ranking_as_signals, les deux sorties sont les rangs (1 = signal le plus élevé)
    assert long_sig.equals(short_sig)
    assert (long_sig.min(axis=1) == 1).all() and (long_sig.max(axis=1) == sig.shape[1]).all()

    strategy = WeightingStrategyFactory.select_weighting_strategy(strategy_type="rank_based_weighting", method="equal")
    long_w, short_w = strategy.compute_weights(long_signals=long_sig, short_signals=short_sig, fix_threshold=3)

    assert ((long_w > 0).sum(axis=1) == 3).all() and ((short_w > 0).sum(axis=1) == 3).all()
    np.testing.assert_allclose(long_w.sum(axis=1), 1.0)
    np.testing.assert_allclose(short_w.sum(axis=1), 1.0)
    # les poids « short » du code 2024 sont positifs (voir README, section Limites)
    assert (short_w >= 0).all().all()

    # la jambe longue prend les signaux les plus élevés, la jambe courte les plus faibles
    for d in sig.index:
        top = set(sig.loc[d].nlargest(3).index)
        bottom = set(sig.loc[d].nsmallest(3).index)
        assert set(long_w.columns[long_w.loc[d] > 0]) == top
        assert set(short_w.columns[short_w.loc[d] > 0]) == bottom
