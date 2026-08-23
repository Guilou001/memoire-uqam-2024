from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class EvaluateModelPerformance:
    def __init__(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                 y_train: pd.DataFrame, metrics: dict[str, dict[str, Any]]) -> None:
        """
        Initialise la classe avec les vraies valeurs, les valeurs prédites, les valeurs en échantillon et les métriques à utiliser.

        :param y_true: Véritables étiquettes.
        :param y_pred: Étiquettes prédites.
        :param y_train: Valeurs en échantillon.
        :param metrics: Dictionnaire des métriques à utiliser et leurs paramètres supplémentaires.
        """
        self._y_true: pd.DataFrame = y_true
        self._y_pred: pd.DataFrame = y_pred
        self._y_train: pd.DataFrame = y_train
        self._metrics: dict[str, dict[str, Any]] = metrics
        self._metric_functions: dict[str, Callable[..., float]] = self._initialize_metric_functions()

    @staticmethod
    def _r_squared_modified(y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series) -> float:
        """
        Calcule le R² ajusté pour les valeurs en échantillon et hors échantillon.

        :param y_true: Véritables étiquettes.
        :param y_pred: Étiquettes prédites.
        :param y_train: Valeurs en échantillon.
        :return: Valeur de R² ajusté.
        """
        SSR = np.sum((y_true - y_pred) ** 2)
        SST = np.sum((y_true - np.mean(y_train)) ** 2)
        return 1 - SSR / SST

    def _initialize_metric_functions(self) -> dict[str, Callable[..., float]]:
        """
        Initialise le dictionnaire des fonctions de métriques.

        :return: Dictionnaire des fonctions de métriques.
        """
        return {
            "accuracy": accuracy_score,
            "f1_score": f1_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "log_loss": log_loss,
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "r2_score": r2_score,
            "r2_score_modified": self._r_squared_modified,
        }

    def _evaluate_metric(self, metric_name: str, y_true_col: pd.Series, y_pred_col: pd.Series,
                         y_train_col: pd.Series, metric_params: dict[str, Any]) -> float:
        """
        Évalue une métrique spécifique.

        :param metric_name: Nom de la métrique.
        :param y_true_col: Véritables étiquettes pour une colonne spécifique.
        :param y_pred_col: Étiquettes prédites pour une colonne spécifique.
        :param y_train_col: Valeurs en échantillon pour une colonne spécifique.
        :param metric_params: Paramètres supplémentaires pour la métrique.
        :return: Valeur de la métrique évaluée.
        """
        metric_function = self._metric_functions.get(metric_name)
        if not metric_function:
            raise ValueError(f"Metric {metric_name} not recognized")

        if metric_name == "r2_score_modified":
            return metric_function(y_true_col, y_pred_col, y_train_col, **metric_params)
        return metric_function(y_true_col, y_pred_col, **metric_params)

    def evaluate(self) -> pd.DataFrame:
        """
        Évalue les performances du modèle selon les métriques spécifiées pour chaque ticker.

        :return: DataFrame des résultats des métriques par ticker.
        """
        results: dict[str, dict[str, float]] = {metric_name: {} for metric_name in self._metrics.keys()}
        tickers = self._y_true.columns

        for ticker in tickers:
            y_true_col = self._y_true[ticker]
            y_pred_col = self._y_pred[ticker]
            y_train_col = self._y_train[ticker]
            for metric_name, metric_params in self._metrics.items():
                results[metric_name][ticker] = self._evaluate_metric(metric_name, y_true_col, y_pred_col, y_train_col, metric_params)

        return pd.DataFrame(results).T

    def compute_mean_metrics_by_long_short(self, long_weights: pd.DataFrame, is_long_only: bool,
                                           metrics_by_ticker: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Compute the mean metrics for long and short tickers.

        :param long_weights: DataFrame des poids des tickers.
        :param is_long_only: Booléen indiquant si seuls les tickers longs doivent être pris en compte.
        :param metrics_by_ticker: DataFrame des résultats de evaluate_and_format_for_each_ticker. Si None, il sera calculé.
        :return: DataFrame des moyennes des métriques pour les tickers longs et, si applicable, courts.
        """
        if metrics_by_ticker is None:
            metrics_by_ticker = self.evaluate()

        tickers = long_weights.columns
        long_tickers = [ticker for ticker in tickers if long_weights[ticker].iloc[0] > 0]
        short_tickers = [ticker for ticker in tickers if long_weights[ticker].iloc[0] == 0]

        available_tickers = metrics_by_ticker.columns.intersection(long_tickers + short_tickers)
        long_tickers = [ticker for ticker in long_tickers if ticker in available_tickers]
        short_tickers = [ticker for ticker in short_tickers if ticker in available_tickers]

        if not long_tickers:
            raise ValueError("No long tickers are available in the results DataFrame.")

        mean_long = metrics_by_ticker[long_tickers].mean(axis=1)

        if is_long_only:
            return mean_long.to_frame(name="mean_long")

        if not short_tickers:
            raise ValueError("No short tickers are available in the results DataFrame.")

        mean_short = metrics_by_ticker[short_tickers].mean(axis=1)
        mean_long_short = (mean_long + mean_short) / 2
        return mean_long_short.to_frame(name="mean_long_short")



if __name__ == '__main__':
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.prediction_pipeline.prediction_pipeline import PredictionPipeline
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.resample_data.data_resampler import DataResampler

    # increase the number of columns to display
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    dr = DataReader()
    relative_data_path = "../../data/raw_data/canadian_stocks_1999-11-01_to_2024-06-01.csv"
    relative_macro_data_path = "../../data/raw_data/macro_data_vif.csv"

    prices_data = dr.read_single_columns_level_data(relative_file_path=relative_data_path, index_col=0)
    macro_data = dr.read_single_columns_level_data(relative_file_path=relative_macro_data_path, index_col=0)

    print(prices_data.head(8))
    print(macro_data.head(8))

    dp = DataPreprocessor()
    prices_data_preprocessed = dp.preprocess(data=prices_data)
    macro_data_preprocessed = dp.preprocess_macro_data(data=macro_data)

    prices_data_aligned, macro_data_aligned = dp.align_dataframes_within_common_period(
        dataframe_1=prices_data_preprocessed,
        dataframe_2=macro_data_preprocessed
    )

    dr = DataResampler()

    prices_data_resampled = dr.resample_and_forward_fill(
        data_to_resample=prices_data_aligned,
        reference_data=macro_data_aligned
    )

    prices_data_resampled = dr.convert_datetime_index_to_period(
        data=prices_data_resampled,
        freq='M'
    )

    macro_data_resampled = dr.convert_datetime_index_to_period(
        data=macro_data_aligned,
        freq='M'
    )

    fptr = FromPricesToReturns(data=prices_data_resampled)
    returns = fptr.compute_returns(return_type='logarithmic')
    macro_data_resampled = macro_data_resampled.iloc[1:]

    # --------- VARIABLES DE TEST --------- #

    # Dictionnaire de pipeline endogène et exogène
    endogenous_pipeline_dict = {}

    exogenous_pipeline_dict = {
        "min_max_scaler": {},
    }

    # Paramètres du forecaster
    # estimator_name = "lasso_regressor"
    # estimator_params = {"alpha": 0.1}

    estimator_name = "lasso_regressor"
    estimator_params = {"alpha": 0.1}
    # estimator_params = {"max_iter": 10000, "learning_rate": 0.1, "max_depth": 3}
    # estimator_params = {"C": 1.0, "kernel": "rbf", "degree": 3, "gamma": "scale"}
    # estimator_params = {"max_iter": 10000, "C": 1.0, "penalty": "l2"}
    make_reduction_params = {"strategy": "recursive", "window_length": 12, "scitype": "tabular-regressor"}

    # ------------------------------------- #

    # in returns dataframe, replace 1 where returns > 0 and 0 otherwise
    # returns_binary = returns.applymap(lambda x: 1 if x > 0 else 0).astype(int)

    # Initialisation et utilisation de la classe PredictionPipelineSktime
    pp = PredictionPipeline(endogenous_data=returns, exogenous_data=macro_data_resampled)
    pp.fit_pipeline(
        endogenous_pipeline_dict=endogenous_pipeline_dict,
        exogenous_pipeline_dict=exogenous_pipeline_dict,
        estimator_name=estimator_name,
        estimator_params=estimator_params,
        make_reduction_params=make_reduction_params
    )
    y_pred = pp.predict()
    y_true = pp.y_test

    # metrics_to_use = {
    #     "accuracy": {},
    #     "f1_score": {"average": "macro"},
    #     "precision_score": {"average": "macro"},
    #     "recall_score": {"average": "macro"},
    #     "log_loss": {},
    # }

    metrics_to_use = {
        "mean_squared_error": {},
        "mean_absolute_error": {},
        "r2_score": {},
    }

    emp = EvaluateModelPerformance(y_true=y_true, y_pred=y_pred, y_train=pp.y_train, metrics=metrics_to_use)
    print(emp.evaluate())

    long_weights = pd.DataFrame(data={"ticker_1": [1], "ticker_2": [0], "ticker_3": [1]})
    print(emp.compute_mean_metrics_by_long_short(long_weights=long_weights, is_long_only=False))
    print(emp.compute_mean_metrics_by_long_short(long_weights=long_weights, is_long_only=True))


