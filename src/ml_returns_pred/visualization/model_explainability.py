import multiprocessing
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

multiprocessing.set_start_method("spawn", force=True)


class ModelExplainability:
    def __init__(self, endogenous_train: pd.DataFrame, exogenous_train: pd.DataFrame):
        """
        Initializes the ModelExplainability class.

        :param endogenous_train: DataFrame containing the endogenous training data.
        :param exogenous_train: DataFrame containing the exogenous training data.
        """
        self.endogenous_train = endogenous_train
        self.exogenous_train = exogenous_train
        self.forecaster: ForecasterAutoregMultiSeries | None = None
        self.ticker_to_level_map = self._map_tickers_to_levels()
        self.X_train_dict: dict[str, pd.DataFrame] = {}  # Stores X_train for each ticker
        self.shap_values_dict: dict[str, np.ndarray] = {}  # Stores SHAP values for each ticker
        self.aggregated_shap_values: np.ndarray | None = None  # Stores aggregated SHAP values

    def _map_tickers_to_levels(self) -> dict[str, int]:
        """
        Creates a dictionary mapping each ticker (column name in endogenous_train)
        to its position in the '_level_skforecast' column.

        :return: Dictionary mapping ticker names to level indices.
        """
        return {ticker: idx for idx, ticker in enumerate(self.endogenous_train.columns)}

    def fit_model(self, forecaster: ForecasterAutoregMultiSeries,
                  store_last_window: Union[bool, list] = True,
                  store_in_sample_residuals: bool = True,
                  suppress_warnings: bool = False) -> ForecasterAutoregMultiSeries:
        """
        Fits the regression model.

        :param forecaster: The ForecasterAutoregMultiSeries model to fit.
        :param store_last_window: Option to store the last window.
        :param store_in_sample_residuals: Option to store in-sample residuals.
        :param suppress_warnings: Option to suppress warnings.
        :return: The fitted ForecasterAutoregMultiSeries model.
        """
        self.forecaster = forecaster
        self.forecaster.fit(
            series=self.endogenous_train,
            exog=self.exogenous_train,
            store_last_window=store_last_window,
            store_in_sample_residuals=store_in_sample_residuals,
            suppress_warnings=suppress_warnings
        )
        return self.forecaster

    def prepare_data_for_explainability(self):
        """
        Prepares data for model explainability by creating X_train for each ticker.
        Filters out tickers with missing values.
        """
        X_train, _ = self.forecaster.create_train_X_y(
            series=self.endogenous_train,
            exog=self.exogenous_train
        )

        self.X_train_dict = {
            ticker: X_train[X_train['_level_skforecast'] == level_value].drop(columns=['_level_skforecast'])
            for ticker, level_value in self.ticker_to_level_map.items()
            if not X_train[X_train['_level_skforecast'] == level_value].isna().any().any()
        }

        excluded_tickers = set(self.ticker_to_level_map.keys()) - set(self.X_train_dict.keys())
        if excluded_tickers:
            print(f"Excluded tickers due to missing values: {excluded_tickers}")

    def compute_shap_values(self, n_jobs: int = -1):
        """
        Calculates SHAP values for the model for each ticker and stores them in a dictionary.
        Filters out SHAP values with NaN or infinite values.

        :param n_jobs: Number of jobs for parallel processing (default: -1 uses all available cores).
        """
        if not self.X_train_dict:
            raise ValueError("Training data not prepared. Call prepare_data_for_explainability first.")

        shap.initjs()
        explainer = shap.TreeExplainer(model=self.forecaster.regressor)

        self.shap_values_dict = dict(Parallel(n_jobs=n_jobs)(
            delayed(self._compute_shap_values_for_ticker)(ticker, explainer)
            for ticker in self.X_train_dict
        ))

        # Filter out tickers with NaN or infinite SHAP values
        self.shap_values_dict = {
            ticker: values for ticker, values in self.shap_values_dict.items()
            if not np.isnan(values).any() and not np.isinf(values).any()
        }

        if len(self.shap_values_dict) < len(self.X_train_dict):
            excluded_tickers = set(self.X_train_dict.keys()) - set(self.shap_values_dict.keys())
            print(f"Excluded tickers due to invalid SHAP values: {excluded_tickers}")

    def _compute_shap_values_for_ticker(self, ticker: str, explainer) -> tuple[str, np.ndarray]:
        """
        Computes SHAP values for a single ticker.

        :param ticker: The ticker for which SHAP values will be computed.
        :param explainer: The SHAP explainer object.
        :return: A tuple containing the ticker and SHAP values for the ticker.
        """
        X_train = self.X_train_dict[ticker]
        shap_values = explainer.shap_values(X=X_train, check_additivity=False)
        return ticker, shap_values

    def aggregate_shap_values(self) -> np.ndarray:
        """
        Aggregates SHAP values across all tickers by averaging, considering only those
        tickers that have the maximum shape.

        :return: Aggregated SHAP values.
        """
        if not self.shap_values_dict:
            raise ValueError("SHAP values have not been calculated yet. Call compute_shap_values first.")

        # Determine the shape of SHAP values for each ticker
        shap_shapes = {ticker: values.shape for ticker, values in self.shap_values_dict.items()}
        shape_counts = {shape: list(shap_shapes.values()).count(shape) for shape in set(shap_shapes.values())}

        # Find the most common shape (max_shape) and keep only tickers with this shape
        max_shape = max(shape_counts, key=shape_counts.get)
        filtered_shap_values_dict = {ticker: values for ticker, values in self.shap_values_dict.items()
                                     if values.shape == max_shape}

        if len(filtered_shap_values_dict) < len(self.shap_values_dict):
            print(f"Filtered out tickers with shapes different from max_shape {max_shape}.")
            print(f"Aggregating SHAP values for {len(filtered_shap_values_dict)} out of {len(self.shap_values_dict)} tickers.")

        # Aggregate SHAP values across the tickers with the max_shape
        self.aggregated_shap_values = np.mean(np.stack(list(filtered_shap_values_dict.values()), axis=0), axis=0)

        return self.aggregated_shap_values

    def plot_summary_shap(self, plot_type: str = "summary", aggregate: bool = True, save_path: str | None = None,
                          show: bool = True):
        """
        Generates and optionally saves SHAP plots (summary or bar) based on the specified plot type.

        :param plot_type: Type of plot to generate ("summary" or "bar").
        :param aggregate: Whether to plot aggregated SHAP values or individual ticker's SHAP values.
        :param save_path: Optional path to save the plot. If None, the plot is not saved.
        :param show: Whether to display the plot.
        """
        shap_values, features = (self.aggregate_shap_values(),
                                 self.X_train_dict[next(iter(self.X_train_dict))]) if aggregate else \
                                (None, None)

        if aggregate:
            self._plot_shap_values(shap_values=shap_values, features=features, plot_type=plot_type,
                                   save_path=save_path, show=show)
        else:
            for ticker in self.shap_values_dict:
                shap_values = self.shap_values_dict[ticker]
                features = self.X_train_dict[ticker]
                self._plot_shap_values(shap_values=shap_values, features=features, plot_type=plot_type,
                                       save_path=save_path, show=show)

    @staticmethod
    def _plot_shap_values(shap_values, features, plot_type: str, save_path: str | None, show: bool = True):
        """
        Helper function to plot and optionally save SHAP values.

        :param shap_values: SHAP values to plot.
        :param features: Features corresponding to the SHAP values.
        :param plot_type: Type of plot to generate ("summary" or "bar").
        :param save_path: Optional path to save the plot. If None, the plot is not saved.
        :param show: Whether to display the plot.
        """
        plt.figure()

        if plot_type == "summary":
            # Use SHAP's summary_plot but without immediate showing
            shap.summary_plot(shap_values=shap_values, features=features, show=False)
        elif plot_type == "bar":
            # Use SHAP's summary_plot with bar type but without immediate showing
            shap.summary_plot(shap_values=shap_values, features=features, plot_type='bar', show=False)

        if save_path:
            plt.savefig(f"{save_path}", bbox_inches='tight')  # Use bbox_inches to ensure everything is saved
        if show:
            plt.show()
        plt.close()

    def main(self):
        """
        Main method to run the explainability process.
        """
        self.prepare_data_for_explainability()
        self.compute_shap_values(n_jobs=-1)
        self.plot_summary_shap(plot_type="summary", aggregate=True, save_path='summary_plot.png')
        self.plot_summary_shap(plot_type="bar", aggregate=True, save_path='bar_plot.png')


if __name__ == '__main__':
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.prediction_pipeline.prediction_pipeline_skforecast import PredictionPipelineSkforecast
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.resample_data.data_resampler import DataResampler

    # Load and preprocess data
    dr = DataReader()
    relative_data_path = "../../data/raw_data/us_stocks_2000-01-01_to_2024-06-01.csv"
    relative_macro_data_path = "../../data/raw_data/Fred-MD.csv"
    prices_data = dr.read_single_columns_level_data(relative_file_path=relative_data_path, index_col=0)
    macro_data = dr.read_single_columns_level_data(relative_file_path=relative_macro_data_path,
                                                   index_col=0, sep=';', parse_dates=['sasdate'])

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

    # Set up pipeline and perform backtesting
    pipeline = PredictionPipelineSkforecast(
        endogenous_data=returns,
        exogenous_data=macro_data_resampled,
    )

    regressor = 'decision_tree_regressor'

    metrics, predictions = pipeline.backtest_model_without_tuning(regressor=regressor,
                                                                  regressor_dict={},
                                                                  lags=12, cutoff_date='2023-01-01',
                                                                  backtest_params={
                                                                      'steps': 1,
                                                                      'metric': ['mean_squared_error'],
                                                                      'fixed_train_size': False,
                                                                      'gap': 0,
                                                                      'skip_folds': None,
                                                                      'allow_incomplete_fold': True,
                                                                      'levels': None,
                                                                      'add_aggregated_metric': True,
                                                                      'refit': True,
                                                                      'interval': None,
                                                                      'n_boot': 500,
                                                                      'random_state': 123,
                                                                      'in_sample_residuals': True,
                                                                      'n_jobs': 'auto',
                                                                      'verbose': False,
                                                                      'show_progress': True,
                                                                      'suppress_warnings': True
                                                                  },
                                                                  forecaster_params={
                                                                      'encoding': 'ordinal_category',
                                                                      'transformer_series': None,
                                                                      'transformer_exog': 'standard_scaler',
                                                                      'weight_func': None,
                                                                      'series_weights': None,
                                                                      'differentiation': None,
                                                                      'dropna_from_series': False,
                                                                      'fit_kwargs': None,
                                                                      'forecaster_id': None
                                                                  })

    print(metrics)
    print(predictions)

    # Explainability process
    me = ModelExplainability(endogenous_train=pipeline.endogenous_train, exogenous_train=pipeline.exogenous_train)

    me.fit_model(forecaster=pipeline.forecaster)
    me.prepare_data_for_explainability()
    me.compute_shap_values(n_jobs=-1)
    me.plot_summary_shap(plot_type="summary", aggregate=True, save_path='summary_plot.png')
    me.plot_summary_shap(plot_type="bar", aggregate=True, save_path='bar_plot.png')
