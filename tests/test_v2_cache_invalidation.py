"""Le cache de prédictions est invalidé quand l'empreinte des données change (revue du 2026-08-23)."""

import pandas as pd

from mlrp import config as cfg
from mlrp.config import RunSpec, TuningSpec
from mlrp.data import synthetic_dataset
from mlrp.runner import get_predictions, load_cached


def test_cache_invalidated_when_data_fingerprint_changes(tmp_path):
    ds = synthetic_dataset(n_months=40, n_assets=8, n_macro=2, seed=3)
    cutoff = str(ds.returns_monthly.index[24].date())
    cfg.PERIODS["synthetic-cache"] = {"cutoff": cutoff, "max_date": ds.max_date}
    try:
        spec = RunSpec(country="canada", period="synthetic-cache", model="ridge_regressor", signal="top10",
                       tuning=TuningSpec(tune=False, lags_default=3))
        pred = get_predictions(spec, ds, cache_dir=tmp_path, n_jobs=1)
        key = spec.prediction_key()

        # même empreinte : le cache sert
        again = load_cached(key, tmp_path, expected_fingerprint=ds.fingerprint())
        assert again is not None
        pd.testing.assert_frame_equal(pred.y_pred, again.y_pred, check_freq=False)

        # empreinte différente (données modifiées) : le cache est refusé
        assert load_cached(key, tmp_path, expected_fingerprint="autre-empreinte") is None

        # sans empreinte attendue : comportement historique, le cache sert
        assert load_cached(key, tmp_path) is not None

        # bout en bout : un jeu de données modifié déclenche le recalcul (résultats différents du cache initial)
        ds2 = synthetic_dataset(n_months=40, n_assets=8, n_macro=2, seed=4)
        assert ds2.fingerprint() != ds.fingerprint()
        pred2 = get_predictions(spec, ds2, cache_dir=tmp_path, n_jobs=1)
        assert not pred2.y_pred.equals(pred.y_pred)
    finally:
        del cfg.PERIODS["synthetic-cache"]
