"""Métriques de mlrp : valeurs connues et cohérence avec le code de 2024 pour les métriques de prédiction."""

import numpy as np
import pandas as pd
import pytest

from ml_returns_pred.prediction_pipeline.METRICS import METRICS as LEGACY_METRICS
from mlrp.metrics import (
    METRICS,
    cagr,
    max_drawdown,
    omega,
    performance_table,
    pesaran_timmermann_stat,
    r2_oos_by_level,
    r_squared_modified,
    sharpe,
)


def test_prediction_metrics_equal_legacy():
    rng = np.random.default_rng(0)
    y_true, y_pred, y_train = rng.normal(size=100), rng.normal(size=100), rng.normal(size=300)
    assert r_squared_modified(y_true, y_pred, y_train) == pytest.approx(
        LEGACY_METRICS["r_squared_modified"](y_true, y_pred, y_train))
    yb, pb = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    assert pesaran_timmermann_stat(yb, pb) == pytest.approx(LEGACY_METRICS["pesaran_timmermann_stat"](yb, pb))
    assert METRICS["pesaran_timmermann_p_value"](yb, pb) == pytest.approx(
        LEGACY_METRICS["pesaran_timmermann_p_value"](yb, pb))


def test_r2_survit_a_un_titre_entre_en_bourse_apres_le_debut():
    # un seul rendement manquant dans l'échantillon d'entraînement rendait la moyenne NaN, donc le
    # R² NaN, sans message : c'est ce qui vidait la colonne r2_oos_pooling du panel américain
    rng = np.random.default_rng(7)
    y_true, y_pred = rng.normal(size=50), rng.normal(size=50)
    y_train = rng.normal(size=200)
    reference = r_squared_modified(y_true, y_pred, y_train)
    y_train_troue = y_train.copy()
    y_train_troue[3] = np.nan
    assert np.isfinite(r_squared_modified(y_true, y_pred, y_train_troue))
    # la moyenne sans le point manquant reste très proche de la moyenne complète
    assert r_squared_modified(y_true, y_pred, y_train_troue) == pytest.approx(reference, abs=0.02)
    # un trou dans y_true ou y_pred est écarté au lieu de contaminer la somme
    y_true_troue, y_pred_troue = y_true.copy(), y_pred.copy()
    y_true_troue[0] = np.nan
    y_pred_troue[1] = np.nan
    assert np.isfinite(r_squared_modified(y_true_troue, y_pred_troue, y_train))
    assert np.isnan(r_squared_modified(np.full(3, np.nan), np.zeros(3), y_train))


def test_cagr_calendar_years():
    idx = pd.bdate_range("2010-01-01", "2019-12-31")
    r = pd.Series(0.0, index=idx)
    r.iloc[-1] = (1 + 1.0) - 1  # double en dix ans civils : TCAC environ 7,2 %
    assert cagr(r) == pytest.approx(2 ** (1 / ((idx[-1] - idx[0]).days / 365.25)) - 1)


def test_sharpe_drawdown_omega():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.001, index=idx)
    assert np.isnan(sharpe(r))                           # écart-type nul
    r2 = pd.Series([0.1, -0.5, 0.2], index=idx[:3])
    assert max_drawdown(r2) == pytest.approx(-0.5)
    assert omega(r2) == pytest.approx(0.3 / 0.5)
    t = performance_table(r2, "x")
    assert set(t.index) >= {"CAGR", "Sharpe", "Volatility", "Max_Drawdown", "Sortino", "Omega_Ratio"}


def test_r2_by_level_has_average_and_pooling():
    rng = np.random.default_rng(3)
    y = pd.DataFrame(rng.normal(size=(60, 3)), index=pd.date_range("2005-01-01", periods=60, freq="MS"),
                     columns=list("abc"))
    pred = pd.DataFrame(rng.normal(size=(24, 3)), index=y.index[-24:], columns=list("abc"))
    s = r2_oos_by_level(y, pred, y.iloc[:36])
    assert {"a", "b", "c", "average", "pooling"} <= set(s.index)
    assert s["average"] == pytest.approx(s[["a", "b", "c"]].mean())
