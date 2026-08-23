"""Le calcul des rendements de stratégie : mode « as_published » (code 2024) contre « corrected ».

Cas synthétique : deux titres, un rééquilibrage le 2020-01-01, puis trois jours. Le titre A monte de
10 % par jour, le titre B baisse de 10 % par jour. La jambe longue tient A, la jambe « short » tient B.
"""

import numpy as np
import pandas as pd
import pytest

from ml_returns_pred.compute_strategy_returns.strategy_returns_calculator import StrategyReturnsCalculator

D2 = pd.Timestamp("2020-01-02")


@pytest.fixture()
def tiny_market():
    dates = pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"A": [100, 100, 110, 121, 133.1], "B": [100, 100, 90, 81, 72.9]}, index=dates)
    long_w = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[dates[1]])
    short_w = pd.DataFrame({"A": [0.0], "B": [1.0]}, index=[dates[1]])  # poids positifs, comme dans le code 2024
    return prices, long_w, short_w


def _run(prices, long_w, short_w, mode, is_long_only=False):
    calc = StrategyReturnsCalculator(long_weights=long_w, short_weights=short_w, prices_data_preprocessed=prices,
                                     transaction_fee=0.0, is_long_only=is_long_only, long_short_mode=mode)
    calc.calculate_drifted_weights()
    return calc.compute_strategy_returns()["Portfolio_Returns"]


def test_as_published_adds_the_short_leg(tiny_market):
    prices, long_w, short_w = tiny_market
    r = _run(prices, long_w, short_w, "as_published")
    # 2020-01-02 : +10 % (A) + (-10 %) (B) = 0 : les deux jambes sont additionnées, le portefeuille est long des deux côtés
    assert r.loc[D2] == pytest.approx(0.0, abs=1e-12)


def test_corrected_subtracts_the_short_leg(tiny_market):
    prices, long_w, short_w = tiny_market
    r = _run(prices, long_w, short_w, "corrected")
    # 2020-01-02 : +10 % (long A) - (-10 %) (short B) = +20 %
    assert r.loc[D2] == pytest.approx(0.20, abs=1e-12)
    assert (r.loc[D2:] > 0).all()


def test_modes_differ_on_a_real_shaped_case(tiny_market):
    prices, long_w, short_w = tiny_market
    a = _run(prices, long_w, short_w, "as_published")
    c = _run(prices, long_w, short_w, "corrected")
    assert not np.allclose(a.loc[D2:], c.loc[D2:])


def test_long_only_is_unchanged_by_mode(tiny_market):
    prices, long_w, short_w = tiny_market
    a = _run(prices, long_w, short_w, "as_published", is_long_only=True)
    c = _run(prices, long_w, short_w, "corrected", is_long_only=True)
    pd.testing.assert_series_equal(a, c)
    assert a.loc[D2] == pytest.approx(0.10, abs=1e-12)


def test_invalid_mode_raises(tiny_market):
    prices, long_w, short_w = tiny_market
    with pytest.raises(ValueError):
        StrategyReturnsCalculator(long_weights=long_w, short_weights=short_w, prices_data_preprocessed=prices,
                                  is_long_only=False, long_short_mode="nope")
