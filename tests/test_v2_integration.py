"""Chaîne complète de mlrp sur un jeu synthétique (sans réseau) : prédictions, cache, portefeuilles, métriques."""

import numpy as np
import pandas as pd
import pytest

from mlrp.config import RunSpec, TuningSpec
from mlrp.data import synthetic_dataset
from mlrp.models import predict, split_index
from mlrp.portfolio import build_weights, strategy_returns
from mlrp.runner import get_predictions, load_cached, run


@pytest.fixture(scope="module")
def ds():
    # 25 titres : avec 6 titres, « top 10 » long et court tiendraient les mêmes titres et le long-short vaudrait 0
    return synthetic_dataset(n_months=48, n_assets=25, n_macro=3, seed=11)


def test_split_index():
    idx = pd.date_range("2010-01-01", periods=12, freq="MS")
    df = pd.DataFrame(np.zeros((12, 1)), index=idx)
    assert split_index(df, "2010-06-30") == 6
    assert split_index(df, "2010-06-01") == 6


def test_predict_ridge_without_tuning(ds):
    cutoff = str(ds.returns_monthly.index[29].date())
    res = predict("ridge_regressor", ds.returns_monthly, ds.exog_monthly, cutoff, TuningSpec(tune=False, lags_default=3))
    assert res.y_pred.shape[1] == ds.returns_monthly.shape[1]
    assert res.y_pred.index[0] > pd.Timestamp(cutoff)
    assert len(res.y_pred) == len(ds.returns_monthly) - res.train_size
    assert "average" in res.metrics_levels.index and np.isfinite(res.y_pred.values).all()


def test_predict_ridge_with_tiny_bayes_search(ds):
    cutoff = str(ds.returns_monthly.index[29].date())
    res = predict("ridge_regressor", ds.returns_monthly, ds.exog_monthly, cutoff,
                  TuningSpec(n_trials=2, lags_grid=(2, 3), lags_default=3, seed=5))
    assert res.tuning_results is not None and len(res.tuning_results) == 2
    assert set(res.best_params) >= {"alpha", "max_iter", "tol", "solver"}


def test_prediction_key_stable_for_default_tuning():
    """L'ajout du champ select_best (2026-08-28) ne doit pas invalider le cache des exécutions historiques."""
    spec = RunSpec(country="usa", period="2008-2024", model="ridge_regressor")
    assert spec.prediction_key() == "1ad6fd14d5d2"  # clé observée dans data/cache_v2 avant l'ajout du champ
    with_fix = RunSpec(country="usa", period="2008-2024", model="ridge_regressor",
                       tuning=TuningSpec(select_best=True))
    assert with_fix.prediction_key() != spec.prediction_key()


def test_select_best_retains_best_trial(ds):
    """Avec select_best, les paramètres retenus sont ceux du meilleur essai au sens de la première métrique
    (par défaut, artefact de 2024 : ligne 0 du tri croissant de skforecast, donc le pire essai pour un R²)."""
    import pytest as pt

    cutoff = str(ds.returns_monthly.index[29].date())
    res = predict("ridge_regressor", ds.returns_monthly, ds.exog_monthly, cutoff,
                  TuningSpec(n_trials=3, lags_grid=(2, 3), lags_default=3, seed=5, select_best=True))
    col = next(c for c in res.tuning_results.columns if c.startswith("r_squared_modified"))
    best_row = res.tuning_results.sort_values(col, ascending=False).iloc[0]
    assert res.best_params["alpha"] == pt.approx(best_row["params"]["alpha"])


def test_cache_roundtrip_and_run(ds, tmp_path):
    spec = RunSpec(country="canada", period="2008-2024", model="ridge_regressor", signal="top10",
                   long_short_mode="corrected", tuning=TuningSpec(tune=False, lags_default=3))
    # on contourne la lecture des fichiers réels en passant le jeu synthétique directement
    ds_mod = ds
    object.__setattr__(spec, "period", "2008-2024")
    cutoff = str(ds_mod.returns_monthly.index[29].date())
    from mlrp import config as cfg

    cfg.PERIODS["synthetic"] = {"cutoff": cutoff, "max_date": ds_mod.max_date}
    spec2 = RunSpec(country="canada", period="synthetic", model="ridge_regressor", signal="top10",
                    long_short_mode="corrected", tuning=TuningSpec(tune=False, lags_default=3))
    pred = get_predictions(spec2, ds_mod, cache_dir=tmp_path, n_jobs=1)
    again = load_cached(spec2.prediction_key(), tmp_path)
    assert again is not None
    pd.testing.assert_frame_equal(pred.y_pred, again.y_pred, check_freq=False)  # parquet ne conserve pas la fréquence

    res = run(spec2, cache_dir=tmp_path, dataset=ds_mod)
    assert np.isfinite(res.performance["Sharpe"]) and res.performance["Gross_Exposure"] == pytest.approx(2.0)
    assert res.r2_levels is not None and "pooling" in res.r2_levels.index

    lw, sw = build_weights(pred.y_pred, "positive", "regressor")
    out = strategy_returns(lw, sw, ds_mod.prices_daily, mode="as_published")
    assert len(out.returns) > 0
    del cfg.PERIODS["synthetic"]
