# Data manipulation
# ==============================================================================
import os

# Warnings configuration
# ==============================================================================
import warnings
from typing import Any

import optuna.pruners
import pandas as pd

# Modelling and Forecasting
# ==============================================================================
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import (
    backtesting_forecaster_multiseries,
    bayesian_search_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)

from ml_returns_pred.prediction_pipeline.ESTIMATORS import ESTIMATORS
from ml_returns_pred.prediction_pipeline.METRICS import METRICS
from ml_returns_pred.prediction_pipeline.TRANSFORMERS import TRANSFORMERS

warnings.filterwarnings('once')
# deactivate runtime warning
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


class PredictionPipelineSkforecast:
    def __init__(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame):
        self.endogenous_data = self._validate_and_convert_index(data=endogenous_data)
        self.exogenous_data = self._validate_and_convert_index(data=exogenous_data)
        self.endogenous_train, self.endogenous_test, self.exogenous_train, self.exogenous_test = None, None, None, None
        self.forecaster = None

    @staticmethod
    def _validate_and_convert_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Valide que l'index du DataFrame est de type datetime, sinon il est converti.

        :param data: Le DataFrame à valider.
        :return: Le DataFrame avec un index datetime.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"L'index du DataFrame ne peut pas être converti en DatetimeIndex: {e}")
        return data

    @staticmethod
    def _get_cutoff_index(data: pd.DataFrame, cutoff_date: str) -> int:
        """
        Calcule l'index de coupure pour la division train/test en fonction de la date de coupure.

        :param data: Le DataFrame à diviser.
        :param cutoff_date: La date de fin du train set en format string.
        :return: L'index de coupure en int.
        """
        cutoff_datetime = pd.to_datetime(cutoff_date)
        return data.index.searchsorted(value=cutoff_datetime, side='right')

    def split_train_test_data(self, cutoff_date: str) -> tuple:
        """
        Divise les données en échantillons d'entraînement et de test en fonction de la date de coupure.

        :param cutoff_date: La date de fin du train set en format string.
        :return: Données d'entraînement et de test pour les séries endogènes et exogènes.
        """
        cutoff_index = self._get_cutoff_index(data=self.endogenous_data, cutoff_date=cutoff_date)

        endogenous_train = self.endogenous_data.iloc[:cutoff_index]
        endogenous_test = self.endogenous_data.iloc[cutoff_index:]

        exogenous_train = self.exogenous_data.iloc[:cutoff_index]
        exogenous_test = self.exogenous_data.iloc[cutoff_index:]

        return endogenous_train, endogenous_test, exogenous_train, exogenous_test

    @staticmethod
    def configure_transformers(forecaster_params: dict, transformers_dict: dict):
        """
        Modifie les dictionnaires contenant une clé 'transformer_exog' en remplaçant la chaîne de caractères
        par la fonction correspondante du dictionnaire TRANSFORMERS.

        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.
        :param transformers_dict: Dictionnaire des transformateurs (ex: TRANSFORMERS).
        """
        if 'transformer_exog' in forecaster_params:
            transformer_str = forecaster_params['transformer_exog']
            if transformer_str in transformers_dict:
                forecaster_params['transformer_exog'] = transformers_dict[transformer_str]()
            else:
                raise ValueError(f"Transformateur '{transformer_str}' non reconnu. "
                                 f"Veuillez choisir parmi: {list(transformers_dict.keys())}")

    def _build_forecaster(self, regressor: str, regressor_dict: dict | None, lags: int,
                          forecaster_params: dict = None) -> None:
        """
        Construit et assigne un modèle de régression pour la prédiction de séries temporelles à l'attribut `self.forecaster`.

        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param forecaster_params: Dictionnaire contenant les paramètres suivants :
            - transformer_series: Transformateurs pour les séries endogènes.
            - transformer_exog: Transformateur pour les séries exogènes.
            - weight_func: Fonction de pondération pour les observations.
            - series_weights: Poids pour chaque série.
            - differentiation: Ordre de différenciation des séries.
            - dropna_from_series: Supprimer les observations manquantes.
            - fit_kwargs: Arguments supplémentaires pour la méthode fit du modèle de régression.
            - forecaster_id: Identifiant du modèle de régression.

        :return: None. Assigne le forecaster à l'attribut `self.forecaster`.
        """
        if forecaster_params is None:
            forecaster_params = {}

        # Configuration des transformateurs exogènes
        self.configure_transformers(forecaster_params=forecaster_params, transformers_dict=TRANSFORMERS)

        self.forecaster = ForecasterAutoregMultiSeries(
            regressor=ESTIMATORS[regressor](**regressor_dict),
            lags=lags,
            **forecaster_params
        )

    def _hyperparameters_tuning_by_grid(self, lags_grid: list, grid_search_params_grid: dict, grid_search_params: dict) \
            -> pd.DataFrame:
        """
        Optimise les hyperparamètres du modèle de régression en utilisant une recherche par grille.

        :param lags_grid: Grille des lags à tester.
        :param grid_search_params_grid: Grille des hyperparamètres à tester.
        :param grid_search_params: Dictionnaire contenant les paramètres suivants :
            - steps: Nombre d'étapes pour prédire.
            - metric: Métrique à optimiser (par exemple 'mean_squared_error').
            - initial_train_size: Nombre d'échantillons dans le split d'entraînement initial.
            - aggregate_metric: Méthode d'agrégation de la métrique ('mean', 'median', etc.).
            - fixed_train_size: Si True, la taille du set d'entraînement ne change pas à chaque itération.
            - gap: Nombre de pas à sauter entre le train et le test.
            - skip_folds: Nombre de folds à sauter pendant la validation croisée.
            - allow_incomplete_fold: Si True, autorise les folds incomplets.
            - levels: Niveau ou niveaux pour lesquels le modèle est optimisé.
            - refit: Si True, réadapte le modèle à chaque itération.
            - return_best: Si True, retourne le modèle ajusté avec les meilleurs paramètres.
            - n_jobs: Nombre de cœurs utilisés pour le calcul ('auto' pour utiliser tous les cœurs disponibles).
            - verbose: Niveau de verbosité.
            - show_progress: Si True, montre la progression de l'optimisation.
            - suppress_warnings: Si True, supprime les avertissements pendant la recherche.
            - output_file: Fichier de sortie pour enregistrer les résultats de l'optimisation.

        :return: DataFrame avec les résultats de l'optimisation des hyperparamètres.
        """

        grid_search_params['initial_train_size'] = len(self.endogenous_train)

        tuning_results = grid_search_forecaster_multiseries(
            forecaster=self.forecaster,
            series=self.endogenous_data,
            exog=self.exogenous_data,
            param_grid=grid_search_params_grid,
            lags_grid=lags_grid,
            **grid_search_params
        )

        return tuning_results

    def _hyperparameters_tuning_by_bayes(self, param_distributions: dict, bayes_search_params: dict,
                                         lags_grid: list[int]) \
            -> tuple[pd.DataFrame, object]:
        """
        Optimise les hyperparamètres du modèle de régression en utilisant une recherche bayésienne.

        :param param_distributions: Distributions des hyperparamètres à tester.
        :param bayes_search_params: Dictionnaire contenant les paramètres pour la recherche bayésienne.
        :param lags_grid: Liste des lags à tester et à ajouter aux paramètres bayésiens.

        :return: DataFrame avec les résultats de l'optimisation des hyperparamètres et l'objet `best_trial`.
        """

        bayes_search_params['initial_train_size'] = len(self.endogenous_train)
        bayes_search_params['kwargs_create_study']['pruner'] = optuna.pruners.SuccessiveHalvingPruner()

        # Add 'lags' to the search space
        param_distributions['lags'] = lags_grid

        def search_space(trial):
            search_space_dict = {}
            for param_name, param_values in param_distributions.items():
                if isinstance(param_values, list):
                    search_space_dict[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    low, high = param_values
                    if isinstance(low, int) and isinstance(high, int):
                        search_space_dict[param_name] = trial.suggest_int(param_name, low, high)
                    elif isinstance(low, float) and isinstance(high, float):
                        search_space_dict[param_name] = trial.suggest_float(param_name, low, high)
                    else:
                        raise ValueError(f"Paramètre non supporté : {param_name} avec les valeurs {param_values}.")
                else:
                    raise ValueError(f"Format de paramètre incorrect pour : {param_name}.")
            return search_space_dict

        tuning_results, best_trial = bayesian_search_forecaster_multiseries(
            forecaster=self.forecaster,
            series=self.endogenous_data,
            exog=self.exogenous_data,
            search_space=search_space,
            **bayes_search_params
        )

        return tuning_results, best_trial

    def _backtest_model(self, backtest_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Effectue un backtesting du modèle de régression multi-séries.

        :param backtest_params: Dictionnaire contenant les paramètres suivants :
            - steps: Nombre d'étapes à prédire.
            - metric: Métrique utilisée pour évaluer les performances du modèle.
            - initial_train_size: Nombre d'échantillons dans le split d'entraînement initial.
            - fixed_train_size: Si True, la taille du set d'entraînement ne change pas mais glisse de `steps` à chaque itération.
            - gap: Nombre de pas à sauter entre la fin de l'entraînement et le début du test.
            - skip_folds: Si spécifié, nombre de plis ou liste des plis à sauter durant la validation croisée.
            - allow_incomplete_fold: Si True, permet des plis incomplets.
            - levels: Niveau ou niveaux de séries à prédire. Si `None`, tous les niveaux sont pris en compte.
            - add_aggregated_metric: Si True, ajoute les métriques agrégées si plusieurs séries sont prédites.
            - refit: Si True, réadapte le modèle à chaque itération. Si un entier est fourni, réadapte toutes les n itérations.
            - interval: Intervalle de confiance des prédictions à estimer, défini par une liste de percentiles.
            - n_boot: Nombre d'itérations de bootstrap pour estimer les intervalles de prédiction.
            - random_state: Seed pour la reproductibilité des résultats.
            - in_sample_residuals: Si True, utilise les résidus en échantillon pour créer des intervalles de prédiction.
            - n_jobs: Nombre de cœurs utilisés pour le calcul parallèle. 'auto' pour utiliser tous les cœurs disponibles.
            - verbose: Niveau de verbosité. Si True, affiche des informations supplémentaires durant le backtesting.
            - show_progress: Si True, montre une barre de progression durant le backtesting.
            - suppress_warnings: Si True, supprime les avertissements durant le backtesting.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        backtest_params['initial_train_size'] = len(self.endogenous_train)

        self.configure_metrics(param_dicts=[backtest_params], metrics_dict=METRICS)

        metrics_level, backtest_predictions = backtesting_forecaster_multiseries(
            forecaster=self.forecaster,
            series=self.endogenous_data,
            exog=self.exogenous_data,
            **backtest_params
        )

        return metrics_level, backtest_predictions

    @staticmethod
    def configure_metrics(param_dicts: list, metrics_dict: dict):
        """
        Modifie les dictionnaires contenant une clé 'metric' en remplaçant les chaînes de caractères
        par les fonctions correspondantes du dictionnaire METRICS.

        :param param_dicts: Liste de dictionnaires à modifier (ex: [backtest_params, grid_search_params]).
        :param metrics_dict: Dictionnaire des métriques (ex: METRICS).
        """
        for param_dict in param_dicts:
            if 'metric' in param_dict:
                param_dict['metric'] = [metrics_dict[metric_str] for metric_str in param_dict['metric']]

    def save_tuning_results(self, tuning_results: pd.DataFrame, relative_path: str) -> None:
        """
        Save the tuning results DataFrame to a specified relative path.

        :param tuning_results: DataFrame containing the results of the hyperparameter tuning.
        :param relative_path: Relative path where the CSV file will be saved.
        """
        try:
            # Convert the relative path to an absolute path
            absolute_path = os.path.abspath(relative_path)

            # Create directories if they do not exist
            # os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

            # Save the DataFrame to the specified path
            tuning_results.to_csv(absolute_path, index=False)
            print(f"Tuning results saved successfully to {absolute_path}")
        except Exception as e:
            print(f"An error occurred while saving the tuning results: {e}")

    # ... (Other methods remain unchanged)

    def backtest_model_with_tuning(self, regressor: Any, regressor_dict: dict | None, lags: int,
                                   lags_grid: list[int],
                                   cutoff_date: str, grid_search_params: dict, grid_search_params_grid: dict,
                                   bayes_search_params: dict, bayes_search_params_grid: dict,
                                   backtest_params: dict, forecaster_params: dict | None = None,
                                   tuning_method: str = 'grid', save_path: str | None = None) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Optimise les hyperparamètres du modèle, entraîne le modèle avec les meilleurs paramètres,
        et effectue un backtesting.

        :param tuning_method: Méthode d'optimisation des hyperparamètres ('grid' ou 'bayes').
        :param save_path: If provided, the tuning results will be saved to this path.
        :param grid_search_params_grid:
        :param lags_grid: Grille de lags à tester lors de l'optimisation d'hyperparamètres.
        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param cutoff_date: Date de fin du train set.
        :param grid_search_params: Dictionnaire contenant les paramètres pour la recherche par grille.
        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.
        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """

        self.configure_metrics(param_dicts=[grid_search_params, bayes_search_params], metrics_dict=METRICS)

        # Split des données
        self.endogenous_train, self.endogenous_test, self.exogenous_train, self.exogenous_test = (
            self.split_train_test_data(cutoff_date=cutoff_date)
        )

        # Construction du forecaster
        self._build_forecaster(regressor=regressor, regressor_dict=regressor_dict, lags=lags,
                               forecaster_params=forecaster_params)

        # Optimisation des hyperparamètres
        if tuning_method == 'grid':
            tuning_results = self._hyperparameters_tuning_by_grid(lags_grid=lags_grid,
                                                                  grid_search_params_grid=grid_search_params_grid,
                                                                  grid_search_params=grid_search_params)
        elif tuning_method == 'bayes':
            tuning_results, _ = self._hyperparameters_tuning_by_bayes(param_distributions=bayes_search_params_grid,
                                                                      bayes_search_params=bayes_search_params,
                                                                      lags_grid=lags_grid)
        else:
            raise ValueError("Invalid tuning method. Choose 'grid' or 'bayes'.")

        # Save the tuning results if a save path is provided
        if save_path:
            self.save_tuning_results(tuning_results=tuning_results, relative_path=save_path)

        # Backtesting avec les meilleurs hyperparamètres
        metrics_level, backtest_predictions = self._backtest_model(backtest_params=backtest_params)

        return metrics_level, backtest_predictions

    def backtest_model_without_tuning(self, regressor: Any, regressor_dict: dict | None, lags: int,
                                      cutoff_date: str, backtest_params: dict, forecaster_params: dict = None) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optimise les hyperparamètres du modèle, entraîne le modèle avec les meilleurs paramètres,
        et effectue un backtesting.

        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param cutoff_date
        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.
        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        # Split des données
        self.endogenous_train, self.endogenous_test, self.exogenous_train, self.exogenous_test = (
            self.split_train_test_data(cutoff_date=cutoff_date)
        )

        # Construction du forecaster
        self._build_forecaster(regressor=regressor, regressor_dict=regressor_dict, lags=lags,
                               forecaster_params=forecaster_params)

        # Backtesting avec les meilleurs hyperparamètres
        metrics_level, backtest_predictions = self._backtest_model(backtest_params=backtest_params)

        return metrics_level, backtest_predictions


if __name__ == '__main__':
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.resample_data.data_resampler import DataResampler

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

    prices_data_resampled = dr.specify_datetime_index_frequency(data=prices_data_resampled, freq='MS')
    macro_data_resampled = dr.specify_datetime_index_frequency(data=macro_data_aligned, freq='MS')

    fptr = FromPricesToReturns()
    returns = fptr.compute_returns(data=prices_data_resampled, return_type='logarithmic')
    macro_data_resampled = macro_data_resampled.iloc[1:]

    lags_grid = [12, 24]
    grid_search_params_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.001],
                               'max_depth': [3, 7], 'subsample': [0.8, 1.0]}

    grid_search_params = {
        'steps': 1,
        'metric': ['r_squared_modified'],
        'aggregate_metric': 'average',
        'fixed_train_size': True,
        'gap': 0,
        'skip_folds': None,
        'allow_incomplete_fold': True,
        'levels': None,
        'refit': False,
        'return_best': True,
        'n_jobs': 'auto',
        'verbose': False,
        'show_progress': True,
        'suppress_warnings': False,
        'output_file': None
    }

    backtest_params = {
        'steps': 1,
        'metric': ['r_squared_modified'],
        'fixed_train_size': False,
        'gap': 0,
        'skip_folds': None,
        'allow_incomplete_fold': True,
        'levels': None,
        'add_aggregated_metric': True,
        'refit': False,
        'interval': None,
        'n_boot': 500,
        'random_state': 123,
        'in_sample_residuals': True,
        'n_jobs': 'auto',
        'verbose': False,
        'show_progress': True,
        'suppress_warnings': True
    }

    forecaster_params = {
        'encoding': 'ordinal_category',
        'transformer_series': None,
        'transformer_exog': "standard_scaler",
        'weight_func': None,
        'series_weights': None,
        'differentiation': None,
        'dropna_from_series': False,
        'fit_kwargs': None,
        'forecaster_id': None
    }

    regressor_dict = {
        'boosting_type': "gbdt",
        'num_leaves': 31,
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 100,
        'subsample_for_bin': 200000,
        'objective': None,  # Peut être une chaîne de caractères ou une fonction objective personnalisée
        'class_weight': None,  # Peut être un dictionnaire ou une chaîne de caractères
        'min_split_gain': 0.0,
        'min_child_weight': 1e-3,
        'min_child_samples': 20,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': None,  # Peut être un entier, un RandomState de NumPy, ou un Generator
        'n_jobs': None,  # Peut être un entier
        'importance_type': "split",  # Options: 'split' ou 'gain'
    }

    bayes_search_params_grid = {
        'n_estimators': (50, 500),  # Range of n_estimators to explore
        'learning_rate': (0.001, 0.1),  # Range of learning rates
        'max_depth': (3, 10),  # Range of max_depth
        'subsample': (0.6, 1.0),  # Range for subsample ratio
        'colsample_bytree': (0.6, 1.0),  # Range for colsample_bytree ratio
        'min_child_samples': (10, 100),  # Minimum number of child samples
        'num_leaves': (31, 128),  # Number of leaves in each tree
    }

    # Example bayesian_search_params dictionary
    bayes_search_params = {
        'steps': 12,  # Number of steps to predict
        'metric': ['r_squared_modified'],  # Metric to optimize
        'aggregate_metric': 'average',  # How to aggregate metrics across multiple series
        'fixed_train_size': True,  # Fixed training size during cross-validation
        'gap': 0,  # No gap between training and testing sets
        'skip_folds': None,  # Not skipping any folds
        'allow_incomplete_fold': True,  # Allow incomplete fold in cross-validation
        'levels': None,  # Optimize for all levels
        'refit': False,  # Refit the model with the best parameters
        'return_best': True,  # Return the best trial's parameters
        'n_trials': 5,  # Number of trials for the Bayesian search
        'random_state': 123,  # Random state for reproducibility
        'n_jobs': 'auto',  # Use all available cores
        'verbose': False,  # Minimal verbosity
        'show_progress': True,  # Show progress bar
        'suppress_warnings': False,  # Display warnings
        'engine': 'optuna',  # Use Optuna for Bayesian optimization
        'kwargs_create_study': {},  # Additional arguments for creating the Optuna study
        'kwargs_study_optimize': {}  # Additional arguments for optimizing the study
    }

    cutoff_date = '2024-01-01'

    # returns_binary = returns.map(lambda x: 1 if x > 0 else 0).astype(int)

    pipeline = PredictionPipelineSkforecast(
        endogenous_data=returns,
        exogenous_data=macro_data_resampled,
    )

    regressor = 'light_gbm_regressor'

    metrics, predictions = pipeline.backtest_model_with_tuning(regressor=regressor, regressor_dict=regressor_dict,
                                                               lags=12, lags_grid=lags_grid,
                                                               cutoff_date=cutoff_date,
                                                               grid_search_params=grid_search_params,
                                                               grid_search_params_grid=grid_search_params_grid,
                                                               bayes_search_params=bayes_search_params,
                                                               bayes_search_params_grid=bayes_search_params_grid,
                                                               backtest_params=backtest_params,
                                                               forecaster_params=forecaster_params,
                                                               tuning_method='bayes',
                                                               save_path='../../data/intermediate_data/hyperparameters_tuned/tuning_results.csv')

    print(metrics)
    print(predictions)

    # metrics, predictions = pipeline.backtest_model_without_tuning(regressor=regressor,
    #                                                               regressor_dict=regressor_dict,
    #                                                               lags=12, cutoff_date=cutoff_date,
    #                                                               backtest_params=backtest_params,
    #                                                               forecaster_params=forecaster_params)
    #
    # print(metrics)
    # print(predictions)


