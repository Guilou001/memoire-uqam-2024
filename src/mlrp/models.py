"""Estimateurs et encapsulation de skforecast (recherche bayésienne puis backtest), réglages du mémoire.

Les estimateurs lourds (xgboost, lightgbm, catboost) sont importés à la demande pour que le paquet se charge
sans eux. Les réglages de ``tune`` et ``backtest`` reproduisent ceux des YAML de 2024 : un pas de prévision,
métrique agrégée en moyenne sur les séries, recherche bayésienne sans réajustement sur l'échantillon de test,
backtest avec réajustement à chaque mois et fenêtre d'entraînement croissante.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import import_module

import numpy as np
import pandas as pd

from mlrp.config import TuningSpec, load_model_space, model_family
from mlrp.metrics import FAMILY_METRICS, METRICS

# Filtres ciblés (jamais un « ignore » global : il masquerait aussi les avertissements utiles ailleurs).
warnings.filterwarnings("ignore", message=".*forecaster will be fit.*")  # LongTrainingWarning de skforecast
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="skforecast")

_ESTIMATOR_PATHS = {
    "ridge_regressor": "sklearn.linear_model:Ridge",
    "linear_regression_regressor": "sklearn.linear_model:LinearRegression",
    "lasso_regressor": "sklearn.linear_model:Lasso",
    "elastic_net_regressor": "sklearn.linear_model:ElasticNet",
    "logistic_regression_classifier": "sklearn.linear_model:LogisticRegression",
    "ada_boost_regressor": "sklearn.ensemble:AdaBoostRegressor",
    "ada_boost_classifier": "sklearn.ensemble:AdaBoostClassifier",
    "extra_trees_regressor": "sklearn.ensemble:ExtraTreesRegressor",
    "extra_trees_classifier": "sklearn.ensemble:ExtraTreesClassifier",
    "random_forest_regressor": "sklearn.ensemble:RandomForestRegressor",
    "random_forest_classifier": "sklearn.ensemble:RandomForestClassifier",
    "gradient_boosting_regressor": "sklearn.ensemble:GradientBoostingRegressor",
    "hist_gradient_boosting_classifier": "sklearn.ensemble:HistGradientBoostingClassifier",
    "hist_gradient_boosting_regressor": "sklearn.ensemble:HistGradientBoostingRegressor",
    "mlp_regressor": "sklearn.neural_network:MLPRegressor",
    "mlp_classifier": "sklearn.neural_network:MLPClassifier",
    "knn_classifier": "sklearn.neighbors:KNeighborsClassifier",
    "xgboost_regressor": "xgboost:XGBRegressor",
    "xgboost_classifier": "xgboost:XGBClassifier",
    "light_gbm_regressor": "lightgbm:LGBMRegressor",
    "catboost_regressor": "catboost:CatBoostRegressor",
    "catboost_classifier": "catboost:CatBoostClassifier",
}


def estimator_class(model: str):
    try:
        module, cls = _ESTIMATOR_PATHS[model].split(":")
    except KeyError as exc:
        raise ValueError(f"modèle inconnu : {model}") from exc
    return getattr(import_module(module), cls)


def make_forecaster(model: str, params: dict, lags: int):
    from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
    from sklearn.preprocessing import StandardScaler

    return ForecasterAutoregMultiSeries(
        regressor=estimator_class(model)(**params), lags=lags, encoding="ordinal_category",
        transformer_series=None, transformer_exog=StandardScaler(), weight_func=None, series_weights=None,
        differentiation=None, dropna_from_series=False, fit_kwargs=None, forecaster_id=None,
    )


def make_search_space(space: dict, lags_grid: tuple[int, ...]):
    """Même règle que 2024 : liste -> catégoriel ; (int, int) -> entier ; (float, float) -> réel ; puis « lags »."""
    dist = dict(space)
    dist["lags"] = list(lags_grid)

    def search_space(trial):
        out = {}
        for name, values in dist.items():
            if isinstance(values, list):
                out[name] = trial.suggest_categorical(name, values)
            elif isinstance(values, tuple) and len(values) == 2:
                low, high = values
                if isinstance(low, int) and isinstance(high, int):
                    out[name] = trial.suggest_int(name, low, high)
                elif isinstance(low, float) and isinstance(high, float):
                    out[name] = trial.suggest_float(name, low, high)
                else:
                    raise ValueError(f"bornes non supportées pour {name} : {values}")
            else:
                raise ValueError(f"format incorrect pour {name} : {values}")
        return out

    return search_space


@dataclass
class PredictionResult:
    y_pred: pd.DataFrame
    metrics_levels: pd.DataFrame
    tuning_results: pd.DataFrame | None
    best_params: dict
    train_size: int


def split_index(returns: pd.DataFrame, cutoff: str) -> int:
    """Nombre de lignes d'entraînement : dates <= cutoff (``searchsorted`` côté droit, comme en 2024)."""
    return int(returns.index.searchsorted(pd.Timestamp(cutoff), side="right"))


def predict(model: str, series: pd.DataFrame, exog: pd.DataFrame, cutoff: str, tuning: TuningSpec,
            n_jobs: int | str = 1) -> PredictionResult:
    """Recherche bayésienne (optionnelle) puis backtest glissant à un pas avec réajustement mensuel."""
    import optuna
    from skforecast.model_selection_multiseries import (
        backtesting_forecaster_multiseries,
        bayesian_search_forecaster_multiseries,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    family = model_family(model)
    metric_names, direction = FAMILY_METRICS[family]
    metrics = [METRICS[m] for m in metric_names]
    params, space = load_model_space(model)
    train_size = split_index(series, cutoff)
    forecaster = make_forecaster(model, params, tuning.lags_default)

    tuning_results, best_params = None, dict(params)
    if tuning.tune:
        tuning_results, _ = bayesian_search_forecaster_multiseries(
            forecaster=forecaster, series=series, exog=exog, search_space=make_search_space(space, tuning.lags_grid),
            steps=1, metric=list(metrics), initial_train_size=train_size, aggregate_metric="average",
            fixed_train_size=True, gap=0, skip_folds=None, allow_incomplete_fold=True, levels=None, refit=False,
            return_best=True, n_trials=tuning.n_trials, random_state=tuning.seed, n_jobs=n_jobs, verbose=False,
            show_progress=False, suppress_warnings=True, engine="optuna",
            kwargs_create_study={"direction": direction, "pruner": optuna.pruners.SuccessiveHalvingPruner()},
            kwargs_study_optimize={},
        )
        best_params = dict(tuning_results.iloc[0]["params"]) if "params" in tuning_results else best_params
        if tuning.select_best and direction == "maximize" and "params" in tuning_results:
            # skforecast trie toujours les essais par la première métrique en ordre CROISSANT et
            # ``return_best`` réajuste le forecaster sur la ligne 0 : quand la métrique est à maximiser
            # (R² des régresseurs), c'est le pire essai. Ici, on resélectionne le meilleur et on
            # reconstruit le forecaster avec ses paramètres avant le backtest.
            col = next(c for c in tuning_results.columns if c.startswith(metric_names[0]))
            best_row = tuning_results.sort_values(col, ascending=False).iloc[0]
            best_params = dict(best_row["params"])
            forecaster = make_forecaster(model, {**params, **best_params}, len(best_row["lags"]))

    metrics_levels, y_pred = backtesting_forecaster_multiseries(
        forecaster=forecaster, series=series, exog=exog, steps=1, metric=list(metrics),
        initial_train_size=train_size, fixed_train_size=False, gap=0, skip_folds=None, allow_incomplete_fold=True,
        levels=None, add_aggregated_metric=True, refit=True, interval=None, n_boot=500, random_state=tuning.seed,
        in_sample_residuals=False, n_jobs=n_jobs, verbose=False, show_progress=False, suppress_warnings=True,
    )
    metrics_levels = metrics_levels.set_index("levels")
    y_pred = y_pred.copy()
    y_pred.index = pd.DatetimeIndex(y_pred.index)
    return PredictionResult(y_pred=y_pred, metrics_levels=metrics_levels, tuning_results=tuning_results,
                            best_params={k: _py(v) for k, v in best_params.items()}, train_size=train_size)


def _py(v):
    if isinstance(v, np.generic):
        return v.item()
    return v
