from sktime.forecasting.compose import make_reduction, ForecastingPipeline, TransformedTargetForecaster
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.lag import Lag
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.panel.pca import PCATransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, HistGradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor, ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from ml_returns_pred.prediction_pipeline.ESTIMATORS import ESTIMATORS
from ml_returns_pred.prediction_pipeline.TRANSFORMERS import TRANSFORMERS
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from typing import Any, Dict
import pandas as pd
import numpy as np
import warnings
import os

# Activer la solution de repli sur CPU pour les opérations non prises en charge par MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

warnings.simplefilter(action='ignore', category=FutureWarning)


class TransformerFactory(object):
    """Factory class to create transformers."""

    @staticmethod
    def create_transformer(transformer_name: str, params: dict):
        transformer_class = TRANSFORMERS.get(transformer_name)
        if transformer_class is None:
            raise ValueError(
                f"Transformer {transformer_name} not found in TRANSFORMERS dictionary: {TRANSFORMERS.keys()}."
            )

        if 'sklearn' in str(transformer_class):
            return TabularToSeriesAdaptor(transformer=transformer_class(**params))
        return transformer_class(**params)


class EstimatorFactory(object):
    """Factory class to create forecasters and regressors."""

    @staticmethod
    def create_estimator(estimator_name: str, estimator_params: dict, make_reduction_params: dict = None):
        estimator_class = ESTIMATORS.get(estimator_name)
        if estimator_class is None:
            raise ValueError(f"Estimator {estimator_name} not found in ESTIMATORS dictionary: {ESTIMATORS.keys()}.")

        if 'sklearn' in str(estimator_class):
            return make_reduction(estimator=estimator_class(**estimator_params), **make_reduction_params)
        return estimator_class(**estimator_params)


class PredictionPipelineSktime(object):
    def __init__(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame, test_size: float = 0.67):
        """
        Initialise la pipeline de prédiction.

        :param endogenous_data: Données endogènes.
        :param exogenous_data: Données exogènes.
        :param test_size: Taille de l'ensemble de test.
        """
        self.endogenous_data = endogenous_data
        self.exogenous_data = exogenous_data
        self.y_train, self.y_test, self.X_train, self.X_test = self._split_train_test_data(
            test_size=test_size
        )
        self.fh = self._compute_forecasting_horizon()
        self.pipeline = None

    def _compute_forecasting_horizon(self) -> ForecastingHorizon:
        """
        Calcule l'horizon de prévision.

        :return: Un objet ForecastingHorizon.
        """
        return ForecastingHorizon(values=np.arange(1, len(self.y_test) + 1))

    def _split_train_test_data(self, test_size: float = 0.67):
        """
        Sépare les données en ensembles d'entraînement et de test.

        :param test_size: Taille de l'ensemble de test.

        :return: y_train, y_test, X_train, X_test
        """
        y_train, y_test, X_train, X_test = temporal_train_test_split(
            y=self.endogenous_data,
            X=self.exogenous_data,
            test_size=test_size
        )
        return y_train, y_test, X_train, X_test

    def fit_pipeline(self, endogenous_pipeline_dict: Dict[str, dict], exogenous_pipeline_dict: Dict[str, dict],
                     estimator_name: str, estimator_params: dict, make_reduction_params: dict = None):
        """
        Ajuste la pipeline de données en fonction de l'estimateur (régression).

        :param endogenous_pipeline_dict: Dictionnaire des étapes de la pipeline endogène.
        :param exogenous_pipeline_dict: Dictionnaire des étapes de la pipeline exogène.
        :param estimator_name: Nom de l'estimateur.
        :param estimator_params: Paramètres de l'estimateur.
        :param make_reduction_params: Paramètres pour la fonction make_reduction (si applicable).
        """
        endogenous_pipeline_dict = endogenous_pipeline_dict or {}
        exogenous_pipeline_dict = exogenous_pipeline_dict or {}

        self._fit_forecasting_pipeline(
            endogenous_pipeline_dict=endogenous_pipeline_dict,
            exogenous_pipeline_dict=exogenous_pipeline_dict,
            forecaster_name=estimator_name,
            forecaster_params=estimator_params,
            make_reduction_params=make_reduction_params
        )

    def _fit_forecasting_pipeline(self, endogenous_pipeline_dict: Dict[str, dict], exogenous_pipeline_dict: Dict[str, dict],
                                  forecaster_name: str, forecaster_params: dict, make_reduction_params: dict):
        """
        Ajuste la pipeline de prévision.

        :param endogenous_pipeline_dict: Dictionnaire des étapes de la pipeline endogène.
        :param exogenous_pipeline_dict: Dictionnaire des étapes de la pipeline exogène.
        :param forecaster_name: Nom du forecaster.
        :param forecaster_params: Paramètres du forecaster.
        :param make_reduction_params: Paramètres pour la fonction make_reduction.
        """
        # Create the endogenous pipeline
        endogenous_steps = [(step_name, TransformerFactory.create_transformer(transformer_name=step_name, params=params))
                            for step_name, params in endogenous_pipeline_dict.items()]

        forecaster = EstimatorFactory.create_estimator(
            estimator_name=forecaster_name,
            estimator_params=forecaster_params,
            make_reduction_params=make_reduction_params
        )
        endogenous_steps.append(("forecaster", forecaster))
        endogenous_pipeline = TransformedTargetForecaster(steps=endogenous_steps)

        # Create the exogenous pipeline
        exogenous_steps: Any = [(step_name, TransformerFactory.create_transformer(transformer_name=step_name, params=params))
                                for step_name, params in exogenous_pipeline_dict.items()]
        exogenous_steps.append(("forecaster", endogenous_pipeline))
        self.pipeline = ForecastingPipeline(steps=exogenous_steps)

        # Fit the exogenous pipeline with endogenous data as target
        self.pipeline.fit(y=self.y_train, X=self.X_train, fh=self.fh)

    def predict(self):
        """
        Prédit les valeurs pour les données de test.

        :return: Les prédictions.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not fitted.")

        return self.pipeline.predict(X=self.X_test, fh=self.fh)


if __name__ == '__main__':
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.resample_data.data_resampler import DataResampler
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns

    dr = DataReader()
    relative_data_path = "../../data/raw_data/canadian_stocks_2000-01-01_to_2024-06-01.csv"
    relative_macro_data_path = "../../data/raw_data/macro_data.csv"
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

    estimator_name = "decision_tree_classifier"
    estimator_params = {}
    # estimator_params = {"max_iter": 10000, "learning_rate": 0.1, "max_depth": 3}
    # estimator_params = {"C": 1.0, "kernel": "rbf", "degree": 3, "gamma": "scale"}
    # estimator_params = {"max_iter": 10000, "C": 1.0, "penalty": "l2", "n_jobs": -1}
    make_reduction_params = {"strategy": "recursive", "window_length": 12, "scitype": "tabular-regressor"}

    # ------------------------------------- #

    # in returns dataframe, replace 1 where returns > 0 and 0 otherwise
    returns_binary = returns.applymap(lambda x: 1 if x > 0 else 0).astype(int)

    # Initialisation et utilisation de la classe PredictionPipelineSktime
    pp = PredictionPipelineSktime(endogenous_data=returns_binary, exogenous_data=macro_data_resampled)
    pp.fit_pipeline(
        endogenous_pipeline_dict=endogenous_pipeline_dict,
        exogenous_pipeline_dict=exogenous_pipeline_dict,
        estimator_name=estimator_name,
        estimator_params=estimator_params,
        make_reduction_params=make_reduction_params
    )
    predictions = pp.predict()

    print(predictions)
