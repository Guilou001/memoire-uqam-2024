"""Importance des variables par SHAP, comme au chapitre 3.4 du mémoire, avec deux différences de calcul.

Le modèle expliqué est le forecaster réajusté sur la période d'entraînement (2000 à 2007) avec les
hyperparamètres retenus par la recherche bayésienne (lus dans le cache de prédictions). Les valeurs SHAP
sont calculées titre par titre sur la matrice d'entraînement de skforecast (retards + variables macro),
puis les observations de tous les titres sont EMPILÉES avant le tracé. Le code de 2024 les MOYENNAIT
position par position entre les titres : deux effets de signes opposés s'annulaient, ce qui produisait des
figures aux valeurs presque nulles (visible sur les figures archivées du volet canadien). L'empilement
conserve chaque observation (date, titre) et colore chaque point par la vraie valeur de la variable.

Explicateurs : ``TreeExplainer`` pour les arbres (Extra Trees, XGBoost, Hist Gradient Boosting),
``LinearExplainer`` pour Ridge et la régression logistique. AdaBoost n'est pas couvert (non supporté par
``TreeExplainer`` ; un explicateur par permutations serait trop coûteux sur cette matrice).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from mlrp.config import CACHE_DIR, COUNTRY_LABELS, MODEL_LABELS, RAW_DIR, RESULTS_DIR, RunSpec
from mlrp.data import binarize, build_dataset
from mlrp.models import make_forecaster, split_index

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPLAINABLE = ("ridge_regressor", "xgboost_regressor", "extra_trees_regressor", "logistic_regression_classifier",
               "xgboost_classifier", "hist_gradient_boosting_classifier", "extra_trees_classifier")
_TREE_MODELS = {"xgboost_regressor", "extra_trees_regressor", "xgboost_classifier",
                "hist_gradient_boosting_classifier", "extra_trees_classifier"}


def _cached_params(spec: RunSpec, cache_dir: Path) -> tuple[dict, int]:
    """Hyperparamètres retenus et retards (lags) lus dans le cache de prédictions de ``mlrp run``."""
    from mlrp.config import load_model_space

    d = cache_dir / spec.prediction_key()
    meta = json.loads((d / "meta.json").read_text())
    base_params, _ = load_model_space(spec.model)   # paramètres fixes du YAML (random_state, enable_categorical, …)
    params = {**dict(base_params), **dict(meta["best_params"])}
    lags = spec.tuning.lags_default
    tuning_file = d / "tuning_results.parquet"
    if tuning_file.exists():
        first = pd.read_parquet(tuning_file).iloc[0]
        if "lags" in first.index:
            lag_list = str(first["lags"]).strip("[]").split()
            lags = len(lag_list)
    return params, lags


def shap_frames(spec: RunSpec, cache_dir: Path = CACHE_DIR, raw_dir: Path = RAW_DIR,
                dataset=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valeurs SHAP empilées (observations de tous les titres) et matrice de variables correspondante."""
    import shap

    ds = dataset or build_dataset(spec.country, spec.max_date, raw_dir)
    series = ds.returns_monthly
    if spec.family == "classifier":
        series = binarize(series)
    train_size = split_index(series, spec.cutoff)
    series_train = series.iloc[:train_size]
    exog_train = ds.exog_monthly.iloc[:train_size]

    params, lags = _cached_params(spec, cache_dir)
    forecaster = make_forecaster(spec.model, params, lags)
    forecaster.fit(series=series_train, exog=exog_train, suppress_warnings=True)

    x_train, _ = forecaster.create_train_X_y(series=series_train, exog=exog_train)
    # l'encodage « ordinal_category » fait du niveau (le titre) une variable du modèle : elle reste dans la
    # matrice pour le calcul (sinon les indices de variables seraient décalés), et sort juste avant le tracé
    x_train = x_train.assign(_level_skforecast=x_train["_level_skforecast"].astype(float))
    levels = {ticker: idx for idx, ticker in enumerate(series_train.columns)}
    if spec.model in _TREE_MODELS:
        explainer = shap.TreeExplainer(model=forecaster.regressor)
    else:
        explainer = shap.LinearExplainer(forecaster.regressor, x_train)
    blocks_x, blocks_s = [], []
    for _ticker, level in levels.items():
        x = x_train[x_train["_level_skforecast"] == level]
        if x.empty or x.isna().any().any():
            continue
        values = explainer.shap_values(x, check_additivity=False) if spec.model in _TREE_MODELS \
            else explainer.shap_values(x)
        values = np.asarray(values)
        if values.ndim == 3:              # classifieurs : une tranche par classe, on garde la classe « hausse »
            values = values[..., -1] if values.shape[-1] == 2 else values[-1]
        if np.isnan(values).any() or np.isinf(values).any():
            continue
        blocks_s.append(pd.DataFrame(values, index=x.index, columns=x.columns))
        blocks_x.append(x)
    shap_df, x_df = pd.concat(blocks_s), pd.concat(blocks_x)
    return shap_df.drop(columns=["_level_skforecast"]), x_df.drop(columns=["_level_skforecast"])


def shap_figures(spec: RunSpec, out_dir: Path = RESULTS_DIR / "figures" / "shap", cache_dir: Path = CACHE_DIR,
                 raw_dir: Path = RAW_DIR, dataset=None, max_display: int = 20) -> list[Path]:
    """Essaim (beeswarm) et classement moyen (bar) pour un pays et un modèle ; PNG dans ``results/v2``."""
    import shap

    shap_df, x_df = shap_frames(spec, cache_dir, raw_dir, dataset)
    out_dir = out_dir / spec.country
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for plot_type, suffix in (("dot", "summary"), ("bar", "bar")):
        plt.figure()
        shap.summary_plot(shap_values=shap_df.values, features=x_df, plot_type=plot_type,
                          max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle(f"{COUNTRY_LABELS.get(spec.country, spec.country)}, "
                     f"{MODEL_LABELS.get(spec.model, spec.model)}, entraînement 2000-2007", fontsize=9, y=1.0)
        out = out_dir / f"{suffix}_{spec.model}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        made.append(out)
    return made
