from abc import ABC, abstractmethod

import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.utils import load_forecaster, save_forecaster

from ml_returns_pred.aggregate_results.results_aggregator import ResultsAggregator
from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
from ml_returns_pred.compute_strategy_returns.strategy_returns_calculator import StrategyReturnsCalculator
from ml_returns_pred.create_long_short_portfolio.long_short_portfolio_creator import (
    LongShortPortfolioCreatorFromRanking,
    LongShortPortfolioCreatorFromSignals,
)
from ml_returns_pred.download_data.data_downloader import DataDownloader
from ml_returns_pred.file_management.file_manager import FileManagerStatic
from ml_returns_pred.file_management.folder_cleaner import FolderCleaner
from ml_returns_pred.prediction_pipeline.prediction_pipeline_skforecast import PredictionPipelineSkforecast
from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
from ml_returns_pred.read_config.config_reader import get_merged_config
from ml_returns_pred.read_data.data_reader import DataReader
from ml_returns_pred.resample_data.data_resampler import DataResampler
from ml_returns_pred.strategy_performance_analysis.strategy_performance_analyzer import StrategyPerformanceAnalyzer
from ml_returns_pred.variable_selection.principal_components import PCABasedVariableSelector
from ml_returns_pred.visualization.model_explainability import ModelExplainability
from ml_returns_pred.weighting.weighting import WeightingStrategyFactory


class ReturnsPredictionPipelineAbstract(ABC):

    prediction_pipeline_config_path: str = '../config/meta_config/prediction_pipeline.yaml'
    regression_pipeline_config_path: str = '../config/meta_config/regression_pipeline.yaml'
    classification_pipeline_config_path: str = '../config/meta_config/classification_pipeline.yaml'
    strategy_config_path: str = '../config/strategy_config/'

    def __init__(self, config: dict, strategy_name: str = None):
        self.config = config
        self.strategy_name = strategy_name

    @abstractmethod
    def download_data(self, **kwargs) -> pd.DataFrame | None:
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def preprocess_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def resample_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def variable_selection(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def compute_returns(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def prediction_pipeline(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ForecasterAutoregMultiSeries]:
        pass

    @abstractmethod
    def create_long_short_portfolio(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def model_explainability(self, **kwargs) -> None:
        pass

    @abstractmethod
    def weighting(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def compute_strategy_returns(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def analyze_strategy_returns(self, **kwargs) -> None:
        pass

    @abstractmethod
    def sensibility_analysis(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def main(self):
        pass

    @staticmethod
    def print_separator(message: str) -> None:
        """
        Prints a stylized separator with the given message centered.

        Args:
            message (str): The message to be displayed within the separator.
        """
        separator_line = "═" * 80
        total_width = 80
        padding_each_side = (total_width - len(message) - 2) // 2  # calculate padding for each side
        extra_padding = (total_width - len(message) - 2) % 2  # for odd-length messages

        # Create the title lines with borders
        title_line_1 = "║" + " " * (total_width - 2) + "║"
        title_line_2 = (
                "║"
                + " " * padding_each_side
                + message
                + " " * (padding_each_side + extra_padding)
                + "║"
        )
        title_line_3 = title_line_1

        # Print the separator with title lines
        print(f"\n{separator_line}\n{title_line_1}\n{title_line_2}\n{title_line_3}\n{separator_line}\n")


class ReturnsPredictionPipeline(ReturnsPredictionPipelineAbstract):

    def download_data(self) -> None:
        download_data_dict = self.config["download_data"]

        if not download_data_dict["download_data"]:
            print("Data download is disabled.")
            return None

        dd = DataDownloader(
            start_date=download_data_dict["start_date"],
            end_date=download_data_dict["end_date"],
            interval=download_data_dict["interval"],
            column_to_keep=download_data_dict["column_to_keep"],
            group_by=download_data_dict["group_by"]
        )

        dd.download_stock_data(
            tickers=download_data_dict["tickers"],
            stocks_file_name=download_data_dict["stocks_file_name"]
        )

        dd.download_benchmark_data(
            benchmark_ticker=download_data_dict["benchmark_ticker"],
            benchmark_file_name=download_data_dict["benchmark_file_name"]
        )

        self.print_separator(message="Data downloaded successfully")

        return None

    def load_data(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        read_data_dict = self.config["read_data"]

        dr = DataReader()
        prices_data = dr.read_single_columns_level_data(
            relative_file_path=read_data_dict["relative_prices_data_path"],
            index_col=0,
        )

        try: #usa macro data
            macro_data = dr.read_single_columns_level_data(
                relative_file_path=read_data_dict["relative_macro_data_path"],
                index_col=0,
                delimiter=";",
                parse_dates=['sasdate'],
            )

        except: #canada macro data
            macro_data = dr.read_single_columns_level_data(
                relative_file_path=read_data_dict["relative_macro_data_path"],
                index_col=0,
                delimiter=",",
                parse_dates=['Date'],
            )

        benchmark_prices = dr.read_single_columns_level_data(
            relative_file_path=read_data_dict["benchmark_prices_relative_path"],
            index_col=0,
            sep=",",
            parse_dates=['Date'],
            date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d")
        )

        self.print_separator(message="Data loaded successfully")

        return prices_data, macro_data, benchmark_prices

    def preprocess_data(self, prices_data: pd.DataFrame, macro_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        data_preprocessor_dict = self.config["preprocess_data"]

        dp = DataPreprocessor()
        prices_data_preprocessed = dp.preprocess(data=prices_data)
        macro_data_preprocessed = dp.preprocess_macro_data(data=macro_data)

        prices_data_aligned, macro_data_aligned = dp.align_dataframes_within_common_period(
            dataframe_1=prices_data_preprocessed,
            dataframe_2=macro_data_preprocessed
        )

        prices_data_aligned = dp.keep_data_until_max_date(
            data=prices_data_aligned,
            max_date=data_preprocessor_dict["max_date"]
        )

        macro_data_aligned = dp.keep_data_until_max_date(
            data=macro_data_aligned,
            max_date=data_preprocessor_dict["max_date"]
        )

        if data_preprocessor_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=prices_data_aligned,
                relative_file_path=
                f"{data_preprocessor_dict['preprocess_data_path']}{self.strategy_name}_prices_data_aligned.csv"
            )
            fm.save_data(
                data=macro_data_aligned,
                relative_file_path=
                f"{data_preprocessor_dict['preprocess_data_path']}{self.strategy_name}_macro_data_aligned.csv"
            )

        self.print_separator(message="Prices data preprocessed successfully")

        return prices_data_aligned, macro_data_aligned

    def resample_data(self, prices_data_aligned: pd.DataFrame, macro_data_aligned: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame]:

        resample_data_dict = self.config["resample_data"]

        dr = DataResampler()

        prices_data_resampled = dr.resample_and_forward_fill(
            data_to_resample=prices_data_aligned,
            reference_data=macro_data_aligned
        )

        prices_data_resampled = dr.specify_datetime_index_frequency(
            data=prices_data_resampled,
            freq=resample_data_dict["frequency"]
        )

        macro_data_resampled = dr.specify_datetime_index_frequency(
            data=macro_data_aligned,
            freq=resample_data_dict["frequency"]
        )

        if resample_data_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=prices_data_resampled,
                relative_file_path=
                f"{resample_data_dict['resample_data_path']}{self.strategy_name}_prices_data_resampled.csv"
            )
            fm.save_data(
                data=macro_data_resampled,
                relative_file_path=
                f"{resample_data_dict['resample_data_path']}{self.strategy_name}_macro_data_resampled.csv"
            )

        self.print_separator(message="Data resampled successfully")

        return prices_data_resampled, macro_data_resampled

    def variable_selection(self, macro_data_resampled: pd.DataFrame) -> pd.DataFrame:
        variable_selection_dict = self.config['variable_selection']

        data_selected = macro_data_resampled

        if variable_selection_dict['use_pca']:
            pca_selector = PCABasedVariableSelector(n_components=variable_selection_dict['n_components'])
            selected_features = pca_selector.fit_transform(data=macro_data_resampled)
            data_selected = pca_selector.get_principal_components()
            pca_selector.plot_explained_variance(save_plot_path=variable_selection_dict['save_plot_path'])

            if variable_selection_dict['save_data']:
                fm = FileManagerStatic()
                fm.save_data(
                    data=selected_features,
                    relative_file_path=f"{variable_selection_dict['variable_selection_path']}"
                )

        self.print_separator(message="Features selected successfully")

        return data_selected

    def compute_returns(self, prices_data_resampled: pd.DataFrame, benchmark_prices: pd.DataFrame) -> pd.DataFrame:
        from_prices_to_returns_dict = self.config['from_prices_to_returns']

        fptr = FromPricesToReturns()
        returns = fptr.compute_returns(
            data=prices_data_resampled,
            return_type=from_prices_to_returns_dict["return_type"],
            binarize=from_prices_to_returns_dict["binarize"],
            fractional_differentiation=from_prices_to_returns_dict["fractional_differentiation"],
        )

        benchmark_returns = fptr.compute_returns(
            data=benchmark_prices,
            return_type=from_prices_to_returns_dict["return_type"],
            binarize=from_prices_to_returns_dict["binarize"],
            fractional_differentiation=from_prices_to_returns_dict["fractional_differentiation"],
        )

        if from_prices_to_returns_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=returns,
                relative_file_path=
                f"{from_prices_to_returns_dict['from_prices_to_returns_path']}{self.strategy_name}_returns.csv"
            )

            fm.save_data(
                data=benchmark_returns,
                relative_file_path=
                f"{from_prices_to_returns_dict['from_prices_to_returns_path']}{benchmark_returns.columns[0]}_returns.csv"
            )

        self.print_separator(message="Returns computed successfully")

        return returns

    def prediction_pipeline(self, returns: pd.DataFrame, macro_data: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ForecasterAutoregMultiSeries]:
        prediction_pipeline_dict = self.config["prediction_pipeline"]

        if prediction_pipeline_dict['use_existing_predictions']:
            fm = FileManagerStatic()
            y_pred = fm.load_data(
                relative_file_path=
                f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_y_pred.csv",
                index_col=0,
                parse_dates=[0],
                date_format="%Y-%m-%d"
            )
            y_train = fm.load_data(
                relative_file_path=
                f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_endogenous_train.csv",
                index_col=0,
                parse_dates=[0],
                date_format="%Y-%m-%d"
            )
            X_train = fm.load_data(
                relative_file_path=
                f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_exogenous_train.csv",
                index_col=0,
                parse_dates=[0],
                date_format="%Y-%m-%d"
            )
            forecaster = load_forecaster(
                file_name=
                f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_forecaster.pkl",
                verbose=True
            )

        else:

            pp_skforecast = PredictionPipelineSkforecast(endogenous_data=returns, exogenous_data=macro_data.iloc[1:])

            if prediction_pipeline_dict['optimize_hyperparameters']:

                metrics_levels, y_pred = pp_skforecast.backtest_model_with_tuning(
                    regressor=prediction_pipeline_dict['estimator_name'],
                    regressor_dict=prediction_pipeline_dict['regressor_dict'],
                    lags=prediction_pipeline_dict['lags'],
                    lags_grid=prediction_pipeline_dict['lags_grid'],
                    cutoff_date=prediction_pipeline_dict['cutoff_date'],
                    grid_search_params=prediction_pipeline_dict['grid_search_params'],
                    grid_search_params_grid=prediction_pipeline_dict['grid_search_params_grid'],
                    backtest_params=prediction_pipeline_dict['backtest_params'],
                    forecaster_params=prediction_pipeline_dict['forecaster_params'],
                    bayes_search_params=prediction_pipeline_dict['bayes_search_params'],
                    bayes_search_params_grid=prediction_pipeline_dict['bayes_search_params_grid'],
                    tuning_method=prediction_pipeline_dict['tuning_method'],
                    save_path=
                    f"{prediction_pipeline_dict['tuning_results_save_path']}{self.strategy_name}_tuning_results.csv"
                )

                forecaster = pp_skforecast.forecaster
                y_train = pp_skforecast.endogenous_train
                X_train = pp_skforecast.exogenous_train

            else:

                metrics_levels, y_pred = pp_skforecast.backtest_model_without_tuning(
                    regressor=prediction_pipeline_dict['estimator_name'],
                    regressor_dict=prediction_pipeline_dict['regressor_dict'],
                    lags=prediction_pipeline_dict['lags'],
                    cutoff_date=prediction_pipeline_dict['cutoff_date'],
                    backtest_params=prediction_pipeline_dict['backtest_params'],
                    forecaster_params=prediction_pipeline_dict['forecaster_params'],
                )

                forecaster = pp_skforecast.forecaster
                y_train = pp_skforecast.endogenous_train
                X_train = pp_skforecast.exogenous_train

            # put column "levels" to index in metrics_level dataframe
            metrics_levels = metrics_levels.set_index('levels')

            if prediction_pipeline_dict['save_data']:
                fm = FileManagerStatic()
                fm.save_data(
                    data=y_pred,
                    relative_file_path=
                    f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_y_pred.csv"
                )
                fm.save_data(
                    data=metrics_levels,
                    relative_file_path=
                    f"{prediction_pipeline_dict['evaluate_model_performance_path']}{self.strategy_name}_metrics_levels.csv"
                )
                save_forecaster(
                    forecaster=pp_skforecast.forecaster,
                    file_name=f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_forecaster.pkl",
                    verbose=False)
                fm.save_data(
                    data=pp_skforecast.endogenous_train,
                    relative_file_path=
                    f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_endogenous_train.csv"
                )
                fm.save_data(
                    data=pp_skforecast.exogenous_train,
                    relative_file_path=
                    f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_exogenous_train.csv"
                )

        print(y_pred.head())

        self.print_separator(message="Predictions computed successfully")

        return y_pred, y_train, X_train, forecaster

    def create_long_short_portfolio(self, signals_at_rebalancing_dates: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        create_long_short_portfolio_dict = self.config['create_long_short_portfolio']

        if create_long_short_portfolio_dict['use_ranking']:
            lspcfr = LongShortPortfolioCreatorFromRanking(signals=signals_at_rebalancing_dates)

            long_signals, short_signals = lspcfr.create_long_short_portfolio(
                ranking_strategy=create_long_short_portfolio_dict['ranking_strategy'],
                ascending=create_long_short_portfolio_dict['ascending'],
                method=create_long_short_portfolio_dict['method'],
                selection_method=create_long_short_portfolio_dict['selection_method'],
                percentile_threshold=create_long_short_portfolio_dict['percentile_threshold'],
                fix_threshold=create_long_short_portfolio_dict['fix_threshold'],
                keep_signal_value=create_long_short_portfolio_dict['keep_signal_value'],
                use_ranking_as_signals=create_long_short_portfolio_dict['use_ranking_as_signals']
            )
        else:
            lspcfs = LongShortPortfolioCreatorFromSignals(signals=signals_at_rebalancing_dates)

            long_signals, short_signals = lspcfs.create_long_short_portfolio(
                keep_signal_value=create_long_short_portfolio_dict['keep_signal_value'],
                transform_binary_classification_to_rank=
                create_long_short_portfolio_dict['transform_binary_classification_to_rank'],
                transform_continuous_to_binary=create_long_short_portfolio_dict['transform_continuous_to_binary'],
            )

        if create_long_short_portfolio_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=long_signals,
                relative_file_path=
                f"{create_long_short_portfolio_dict['create_long_short_portfolio_path']}"
                f"{self.strategy_name}_long_signals.csv"
            )
            fm.save_data(
                data=short_signals,
                relative_file_path=
                f"{create_long_short_portfolio_dict['create_long_short_portfolio_path']}"
                f"{self.strategy_name}_short_signals.csv"
            )

        print(long_signals.head())

        self.print_separator(message="Long and short portfolios created successfully")

        return long_signals, short_signals

    def weighting(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        weighting_dict = self.config['weighting']
        create_long_short_portfolio_dict = self.config['create_long_short_portfolio']

        weighting_strategy = WeightingStrategyFactory.select_weighting_strategy(
            strategy_type=weighting_dict['strategy_type'],
            method=weighting_dict['method'],
        )

        long_weights, short_weights = weighting_strategy.compute_weights(
            long_signals=long_signals,
            short_signals=short_signals,
            fix_threshold=create_long_short_portfolio_dict['fix_threshold'],
        )

        if weighting_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=long_weights,
                relative_file_path=
                f"{weighting_dict['weighting_path']}{self.strategy_name}_long_weights.csv"
            )
            fm.save_data(
                data=short_weights,
                relative_file_path=
                f"{weighting_dict['weighting_path']}{self.strategy_name}_short_weights.csv"
            )

        print(long_weights.head())
        print(short_weights.head())

        self.print_separator(message="Weights computed successfully")

        return long_weights, short_weights

    def model_explainability(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                             forecaster: ForecasterAutoregMultiSeries) -> None:
        """
        Permet d'expliquer les résultats du modèle d'arbre de décision.

        :return: None
        """
        model_explainability_dict = self.config['model_explainability']
        prediction_pipeline_dict = self.config['prediction_pipeline']

        model_explainability = ModelExplainability(
            endogenous_train=y_train,
            exogenous_train=X_train,
        )

        if not prediction_pipeline_dict["backtest_params"]["refit"]:

            try:
                model_explainability.fit_model(
                    forecaster=forecaster,
                    store_last_window=model_explainability_dict['store_last_window'],
                    store_in_sample_residuals=model_explainability_dict['store_in_sample_residuals'],
                    suppress_warnings=model_explainability_dict['suppress_warnings'],
                )

                model_explainability.prepare_data_for_explainability()
                model_explainability.compute_shap_values(n_jobs=-1)

                # Plots the SHAP summary plot
                model_explainability.plot_summary_shap(
                    plot_type="summary",
                    aggregate=model_explainability_dict['aggregate'],
                    save_path=f"{model_explainability_dict['summary_plot_save_path']}summary_plot_{self.strategy_name}.png"
                )

                # Plots the SHAP bar plot
                model_explainability.plot_summary_shap(
                    plot_type="bar",
                    aggregate=model_explainability_dict['aggregate'],
                    save_path=f"{model_explainability_dict['summary_plot_save_path']}bar_plot_{self.strategy_name}.png"
                )

                self.print_separator(message="Model Explainability plotted successfully")

            except Exception:
                print(f"Le modèle {forecaster.regressor.__class__.__name__} n'est pas compatible avec SHAP")

    def compute_strategy_returns(self, long_weights: pd.DataFrame, short_weights: pd.DataFrame,
                                 prices_data_preprocessed: pd.DataFrame) -> pd.DataFrame:
        compute_strategy_returns_dict = self.config['compute_strategy_returns']

        strategy_returns_calculator = StrategyReturnsCalculator(
            long_weights=long_weights,
            short_weights=short_weights,
            prices_data_preprocessed=prices_data_preprocessed,
            transaction_fee=compute_strategy_returns_dict['transaction_fee'],
            is_long_only=compute_strategy_returns_dict['is_long_only'],
            long_short_mode=compute_strategy_returns_dict.get('long_short_mode', 'as_published'),
        )

        strategy_returns_calculator.calculate_drifted_weights()
        strategy_returns = strategy_returns_calculator.compute_strategy_returns()

        if compute_strategy_returns_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=strategy_returns,
                relative_file_path=
                f"{compute_strategy_returns_dict['compute_strategy_returns_path']}"
                f"{self.strategy_name}_strategy_returns.csv"
            )

        print(strategy_returns.head())

        self.print_separator(message="Strategy returns computed successfully")

        return strategy_returns

    def analyze_strategy_returns(self, strategy_returns: pd.DataFrame, benchmark_prices: pd.DataFrame = None) -> None:
        analyze_strategy_returns_dict = self.config['analyze_strategy_returns']
        preprocess_data_dict = self.config['preprocess_data']
        prediction_pipeline_dict = self.config['prediction_pipeline']

        # keep benchmark_prices from first date of strategy_returns to the end
        if benchmark_prices is not None:
            benchmark_prices = benchmark_prices.loc[strategy_returns.index[0]:preprocess_data_dict['max_date']]

        strategy_performance_analyzer = StrategyPerformanceAnalyzer(
            portfolio_returns=strategy_returns,
            benchmark_prices=benchmark_prices,
            strategy_name=analyze_strategy_returns_dict['strategy_name']
        )

        strategy_performance_analyzer.generate_backtesting_report_html(
            rf=analyze_strategy_returns_dict['rf'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
            grayscale=analyze_strategy_returns_dict['grayscale'],
            output=analyze_strategy_returns_dict['output'],
            match_dates=analyze_strategy_returns_dict['match_dates'],
        )

        portfolio_key_metrics = strategy_performance_analyzer.get_key_performance_metrics(
            rf=analyze_strategy_returns_dict['rf'],
            annualize=analyze_strategy_returns_dict['annualize'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
        )

        benchmark_key_metrics = strategy_performance_analyzer.get_key_performance_metrics_benchmark(
            benchmark_returns=strategy_performance_analyzer.benchmark_returns.loc[strategy_returns.index[0]:preprocess_data_dict['max_date']],
            rf=analyze_strategy_returns_dict['rf'],
            annualize=analyze_strategy_returns_dict['annualize'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
        )

        if analyze_strategy_returns_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=portfolio_key_metrics,
                relative_file_path=
                f"{analyze_strategy_returns_dict['analyze_strategy_returns_path']}"
                f"{self.strategy_name}_portfolio_key_metrics.csv"
            )
            fm.save_data(
                data=benchmark_key_metrics,
                relative_file_path=
                f"{analyze_strategy_returns_dict['analyze_strategy_returns_path']}"
                f"{benchmark_prices.columns[0]}_key_metrics.csv"
            )

        self.print_separator(message="Strategy returns analyzed successfully")

        return None

    def sensibility_analysis(self, **kwargs) -> pd.DataFrame:
        pass

    def main(self):
        self.download_data()
        prices_data, macro_data, benchmark_prices = self.load_data()
        prices_data_aligned, macro_data_aligned = self.preprocess_data(prices_data=prices_data, macro_data=macro_data)
        prices_data_resampled, macro_data_resampled = self.resample_data(
            prices_data_aligned=prices_data_aligned,
            macro_data_aligned=macro_data_aligned
        )
        macro_data_selected = self.variable_selection(macro_data_resampled=macro_data_resampled)

        returns = self.compute_returns(prices_data_resampled=prices_data_resampled, benchmark_prices=benchmark_prices)
        y_pred, y_train, X_train, forecaster = self.prediction_pipeline(returns=returns, macro_data=macro_data_selected)
        long_signals, short_signals = self.create_long_short_portfolio(signals_at_rebalancing_dates=y_pred)
        long_weights, short_weights = self.weighting(long_signals=long_signals, short_signals=short_signals)

        self.model_explainability(X_train=X_train, y_train=y_train, forecaster=forecaster)

        strategy_returns = self.compute_strategy_returns(
            long_weights=long_weights,
            short_weights=short_weights,
            prices_data_preprocessed=prices_data_aligned
        )
        self.analyze_strategy_returns(strategy_returns=strategy_returns, benchmark_prices=benchmark_prices)


class LassoRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "lasso_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class GradientBoostingRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "gradient_boosting_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class LogisticRegressionClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "logistic_regression_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class HistGradientBoostingClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "hist_gradient_boosting_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class AdaBoostClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "ada_boost_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class KNNClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "knn_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class AdaBoostRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "ada_boost_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class RidgeRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "ridge_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class ElasticNetRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "elastic_net_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class RandomForestRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "random_forest_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class RandomForestClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "random_forest_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class ExtraTreesRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "extra_trees_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class ExtraTreesClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "extra_trees_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class NeuralForecasterLSTMPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "neural_forecaster_lstm"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class MLPRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "mlp_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class MLPClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "mlp_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class GaussianProcessRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "gaussian_process_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class LarsRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "lars_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class LarsLassoRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "lars_lasso_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class LightGBMRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "light_gbm_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class XGBoostRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "xgboost_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class CatBoostRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "catboost_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class XGBoostClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "xgboost_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class CatBoostClassifierPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "catboost_classifier"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.classification_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class ARDRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "ard_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class OMPRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "omp_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class LinearRegressionRegressorPipeline(ReturnsPredictionPipeline):
    def __init__(self):
        self.strategy_name = "linear_regression_regressor"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.regression_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class EquallyWeightedPipeline(ReturnsPredictionPipelineAbstract):
    def __init__(self):
        self.strategy_name = "equally_weighted"
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)

    def download_data(self, **kwargs) -> pd.DataFrame | None:
        download_data_dict = self.config["download_data"]

        if not download_data_dict["download_data"]:
            print("Data download is disabled.")
            return None

        dd = DataDownloader(
            start_date=download_data_dict["start_date"],
            end_date=download_data_dict["end_date"],
            interval=download_data_dict["interval"],
            column_to_keep=download_data_dict["column_to_keep"],
            group_by=download_data_dict["group_by"]
        )

        dd.download_stock_data(
            tickers=download_data_dict["tickers"],
            stocks_file_name=download_data_dict["stocks_file_name"]
        )

        dd.download_benchmark_data(
            benchmark_ticker=download_data_dict["benchmark_ticker"],
            benchmark_file_name=download_data_dict["benchmark_file_name"]
        )

        self.print_separator(message="Data downloaded successfully")

    def load_data(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        read_data_dict = self.config["read_data"]

        dr = DataReader()
        prices_data = dr.read_single_columns_level_data(
            relative_file_path=read_data_dict["relative_prices_data_path"],
            index_col=0,
        )

        benchmark_prices = dr.read_single_columns_level_data(
            relative_file_path=read_data_dict["benchmark_prices_relative_path"],
            index_col=0,
            sep=",",
            parse_dates=['Date'],
            date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d")
        )

        self.print_separator(message="Data loaded successfully")

        return prices_data, benchmark_prices

    def preprocess_data(self, prices_data: pd.DataFrame) -> pd.DataFrame:
        data_preprocessor_dict = self.config["preprocess_data"]

        dp = DataPreprocessor()
        prices_data_preprocessed = dp.preprocess(data=prices_data)

        prices_data_aligned = dp.keep_data_until_max_date(
            data=prices_data_preprocessed,
            max_date=data_preprocessor_dict["max_date"]
        )
        if data_preprocessor_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=prices_data_aligned,
                relative_file_path=
                f"{data_preprocessor_dict['preprocess_data_path']}{self.strategy_name}_prices_data_aligned.csv"
            )

        self.print_separator(message="Prices data preprocessed successfully")

        return prices_data_aligned

    def resample_data(self, prices_data_aligned: pd.DataFrame) -> pd.DataFrame:
        resample_data_dict = self.config["resample_data"]

        dr = DataResampler()

        prices_data_resampled = dr.specify_datetime_index_frequency(
            data=prices_data_aligned,
            freq=resample_data_dict["frequency"]
        )

        if resample_data_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=prices_data_resampled,
                relative_file_path=
                f"{resample_data_dict['resample_data_path']}{self.strategy_name}_prices_data_resampled.csv"
            )

        return prices_data_resampled

    def variable_selection(self, **kwargs) -> pd.DataFrame:
        pass

    def compute_returns(self, prices_data_resampled: pd.DataFrame, benchmark_prices: pd.DataFrame) -> pd.DataFrame:
        from_prices_to_returns_dict = self.config['from_prices_to_returns']

        fptr = FromPricesToReturns()
        returns = fptr.compute_returns(
            data=prices_data_resampled,
            return_type=from_prices_to_returns_dict["return_type"],
            binarize=from_prices_to_returns_dict["binarize"],
            fractional_differentiation=from_prices_to_returns_dict["fractional_differentiation"],
        )

        benchmark_returns = fptr.compute_returns(
            data=benchmark_prices,
            return_type=from_prices_to_returns_dict["return_type"],
            binarize=from_prices_to_returns_dict["binarize"],
            fractional_differentiation=from_prices_to_returns_dict["fractional_differentiation"],
        )

        if from_prices_to_returns_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=returns,
                relative_file_path=
                f"{from_prices_to_returns_dict['from_prices_to_returns_path']}equity_returns.csv"
            )

            fm.save_data(
                data=benchmark_returns,
                relative_file_path=
                f"{from_prices_to_returns_dict['benchmark_returns_path']}{benchmark_returns.columns[0]}_returns.csv"
            )

        self.print_separator(message="Returns computed successfully")

        return returns

    def prediction_pipeline(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ForecasterAutoregMultiSeries]:
        pass

    @staticmethod
    def compute_signals_at_rebalancing_dates(returns: pd.DataFrame):

        signals_at_rebalancing_dates = returns.copy()
        signals_at_rebalancing_dates.loc[:, :] = 1

        return signals_at_rebalancing_dates

    def create_long_short_portfolio(self, signals_at_rebalancing_dates: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        create_long_short_portfolio_dict = self.config['create_long_short_portfolio']

        if create_long_short_portfolio_dict['use_ranking']:
            lspcfr = LongShortPortfolioCreatorFromRanking(signals=signals_at_rebalancing_dates)

            long_signals, short_signals = lspcfr.create_long_short_portfolio(
                ranking_strategy=create_long_short_portfolio_dict['ranking_strategy'],
                ascending=create_long_short_portfolio_dict['ascending'],
                method=create_long_short_portfolio_dict['method'],
                selection_method=create_long_short_portfolio_dict['selection_method'],
                percentile_threshold=create_long_short_portfolio_dict['percentile_threshold'],
                fix_threshold=create_long_short_portfolio_dict['fix_threshold'],
            )
        else:
            lspcfs = LongShortPortfolioCreatorFromSignals(signals=signals_at_rebalancing_dates)

            long_signals, short_signals = lspcfs.create_long_short_portfolio(
                keep_signal_value=create_long_short_portfolio_dict['keep_signal_value'],
                transform_binary_classification_to_rank=
                create_long_short_portfolio_dict['transform_binary_classification_to_rank'],
                transform_continuous_to_binary=create_long_short_portfolio_dict['transform_continuous_to_binary']

            )

        if create_long_short_portfolio_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=long_signals,
                relative_file_path=
                f"{create_long_short_portfolio_dict['create_long_short_portfolio_path']}"
                f"{self.strategy_name}_long_signals.csv"
            )
            fm.save_data(
                data=short_signals,
                relative_file_path=
                f"{create_long_short_portfolio_dict['create_long_short_portfolio_path']}"
                f"{self.strategy_name}_short_signals.csv"
            )

        self.print_separator(message="Long and short portfolios created successfully")

        return long_signals, short_signals

    def model_explainability(self, **kwargs) -> None:
        pass

    def weighting(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        weighting_dict = self.config['weighting']
        create_long_short_portfolio_dict = self.config['create_long_short_portfolio']

        weighting_strategy = WeightingStrategyFactory.select_weighting_strategy(
            strategy_type=weighting_dict['strategy_type']
        )

        long_weights, short_weights = weighting_strategy.compute_weights(
            long_signals=long_signals,
            short_signals=short_signals,
            fix_threshold=create_long_short_portfolio_dict['fix_threshold']
        )

        if weighting_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=long_weights,
                relative_file_path=
                f"{weighting_dict['weighting_path']}{self.strategy_name}_long_weights.csv"
            )
            fm.save_data(
                data=short_weights,
                relative_file_path=
                f"{weighting_dict['weighting_path']}{self.strategy_name}_short_weights.csv"
            )

        self.print_separator(message="Weights computed successfully")

        return long_weights, short_weights

    def compute_strategy_returns(self, long_weights: pd.DataFrame, short_weights: pd.DataFrame,
                                 prices_data_preprocessed: pd.DataFrame) -> pd.DataFrame:
        compute_strategy_returns_dict = self.config['compute_strategy_returns']

        strategy_returns_calculator = StrategyReturnsCalculator(
            long_weights=long_weights,
            short_weights=short_weights,
            prices_data_preprocessed=prices_data_preprocessed,
            transaction_fee=compute_strategy_returns_dict['transaction_fee'],
            is_long_only=compute_strategy_returns_dict['is_long_only'],
            long_short_mode=compute_strategy_returns_dict.get('long_short_mode', 'as_published'),
        )

        strategy_returns_calculator.calculate_drifted_weights()
        strategy_returns = strategy_returns_calculator.compute_strategy_returns()

        if compute_strategy_returns_dict['save_data']:
            strategy_returns.columns = ['Equally Weighted']
            fm = FileManagerStatic()
            fm.save_data(
                data=strategy_returns,
                relative_file_path=
                f"{compute_strategy_returns_dict['benchmark_returns_path']}"
                f"{self.strategy_name}_strategy_returns.csv"
            )

        self.print_separator(message="Strategy returns computed successfully")

        return strategy_returns

    def analyze_strategy_returns(self, strategy_returns: pd.DataFrame, benchmark_prices: pd.DataFrame = None) -> None:
        analyze_strategy_returns_dict = self.config['analyze_strategy_returns']
        prediction_pipeline_dict = self.config['prediction_pipeline']
        preprocess_data_dict = self.config['preprocess_data']

        # keep strategy returns and benchmark_prices from cutoff date to the end
        strategy_returns = strategy_returns.loc[prediction_pipeline_dict['cutoff_date']:preprocess_data_dict['max_date']]
        # compute arithmetic returns from benchmark_prices
        benchmark_returns = benchmark_prices.pct_change().loc[prediction_pipeline_dict['cutoff_date']:preprocess_data_dict['max_date']].squeeze()

        strategy_performance_analyzer = StrategyPerformanceAnalyzer(
            portfolio_returns=strategy_returns,
            benchmark_prices=benchmark_prices,
            strategy_name=analyze_strategy_returns_dict['strategy_name']
        )

        strategy_performance_analyzer.generate_backtesting_report_html(
            rf=analyze_strategy_returns_dict['rf'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
            grayscale=analyze_strategy_returns_dict['grayscale'],
            output=analyze_strategy_returns_dict['output'],
            match_dates=analyze_strategy_returns_dict['match_dates'],
        )

        portfolio_key_metrics = strategy_performance_analyzer.get_key_performance_metrics(
            rf=analyze_strategy_returns_dict['rf'],
            annualize=analyze_strategy_returns_dict['annualize'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
        )

        benchmark_key_metrics = strategy_performance_analyzer.get_key_performance_metrics_benchmark(
            benchmark_returns=benchmark_returns,
            rf=analyze_strategy_returns_dict['rf'],
            annualize=analyze_strategy_returns_dict['annualize'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
        )

        if analyze_strategy_returns_dict['save_data']:
            fm = FileManagerStatic()
            fm.save_data(
                data=portfolio_key_metrics,
                relative_file_path=
                f"{analyze_strategy_returns_dict['analyze_strategy_returns_path']}"
                f"{self.strategy_name}_portfolio_key_metrics.csv"
            )

            fm.save_data(
                data=benchmark_key_metrics,
                relative_file_path=
                f"{analyze_strategy_returns_dict['analyze_strategy_returns_path']}"
                f"{benchmark_prices.columns[0]}_key_metrics.csv"
            )

        self.print_separator(message="Strategy returns analyzed successfully")

        return None

    def sensibility_analysis(self, **kwargs) -> pd.DataFrame:
        pass

    def main(self):
        self.download_data()
        prices_data, benchmark_prices = self.load_data()
        prices_data_aligned = self.preprocess_data(prices_data=prices_data)
        prices_data_resampled = self.resample_data(
            prices_data_aligned=prices_data_aligned
        )
        returns = self.compute_returns(prices_data_resampled=prices_data_resampled, benchmark_prices=benchmark_prices)
        signals_at_rebalancing_dates = self.compute_signals_at_rebalancing_dates(returns=returns)
        long_signals, short_signals = self.create_long_short_portfolio(
            signals_at_rebalancing_dates=signals_at_rebalancing_dates)
        long_weights, short_weights = self.weighting(long_signals=long_signals, short_signals=short_signals)
        strategy_returns = self.compute_strategy_returns(
            long_weights=long_weights,
            short_weights=short_weights,
            prices_data_preprocessed=prices_data_aligned
        )
        self.analyze_strategy_returns(strategy_returns=strategy_returns, benchmark_prices=benchmark_prices)


if __name__ == '__main__':
    # ToDo: Réfléchir à implémenter la variable réalisé, comment jouer sur le mt et sigma_t et ensuite avec un ensemble de titres

    config_path = '../config/meta_config/prediction_pipeline.yaml'
    config = get_merged_config(config_paths=[config_path], strategy_name="Cleaning Folder")

    if config["folder_cleaner"]["clean_data"]:

        folder_cleaner = FolderCleaner(
            folder_paths=config['folder_cleaner']['folder_paths'],
            extensions=config['folder_cleaner']['extensions']
        )
        folder_cleaner.clean_folders()

    # ----------------------------- RUN PIPELINES -----------------------------#

    # linear_regression_regressor_pipeline = LinearRegressionRegressorPipeline()
    # linear_regression_regressor_pipeline.main()

    equally_weighted_pipeline = EquallyWeightedPipeline()
    equally_weighted_pipeline.main()

    # omp_regressor = OMPRegressorPipeline()
    # omp_regressor.main()
    #
    # light_gbm_regressor_pipeline = LightGBMRegressorPipeline()
    # light_gbm_regressor_pipeline.main()
    #
    xgboost_regressor_pipeline = XGBoostRegressorPipeline()
    xgboost_regressor_pipeline.main()

    xgboost_classifier_pipeline = XGBoostClassifierPipeline()
    xgboost_classifier_pipeline.main()

    ridge_regressor_pipeline = RidgeRegressorPipeline()
    ridge_regressor_pipeline.main()

    # gradient_boosting_pipeline = GradientBoostingRegressorPipeline()
    # gradient_boosting_pipeline.main()
    #
    # random_forest_regressor_pipeline = RandomForestRegressorPipeline()
    # random_forest_regressor_pipeline.main()
    #
    ada_boost_regressor_pipeline = AdaBoostRegressorPipeline()
    ada_boost_regressor_pipeline.main()

    extra_trees_regressor_pipeline = ExtraTreesRegressorPipeline()
    extra_trees_regressor_pipeline.main()

    logistic_regression_pipeline = LogisticRegressionClassifierPipeline()
    logistic_regression_pipeline.main()

    hist_gradient_boosting_pipeline = HistGradientBoostingClassifierPipeline()
    hist_gradient_boosting_pipeline.main()

    # ada_boost_classifier_pipeline = AdaBoostClassifierPipeline()
    # ada_boost_classifier_pipeline.main()
    #
    # random_forest_classifier_pipeline = RandomForestClassifierPipeline()
    # random_forest_classifier_pipeline.main()
    #
    extra_trees_classifier_pipeline = ExtraTreesClassifierPipeline()
    extra_trees_classifier_pipeline.main()
    #
    # mlp_classifier_pipeline = MLPClassifierPipeline()
    # mlp_classifier_pipeline.main()

    # ----------------------------- MODELS CEMETERY (RIP) -----------------------------#

    # elastic_net_regressor_pipeline = ElasticNetRegressorPipeline()
    # elastic_net_regressor_pipeline.main()

    # ard_regressor = ARDRegressorPipeline()
    # ard_regressor.main()

    # lasso_pipeline = LassoRegressorPipeline()
    # lasso_pipeline.main()

    # knn_classifier_pipeline = KNNClassifierPipeline()
    # knn_classifier_pipeline.main()

    # gaussian_process_regressor_pipeline = GaussianProcessRegressorPipeline()
    # gaussian_process_regressor_pipeline.main()

    # lars_regressor_pipeline = LarsRegressorPipeline()
    # lars_regressor_pipeline.main()
    #
    # lars_lasso_regressor_pipeline = LarsLassoRegressorPipeline()
    # lars_lasso_regressor_pipeline.main()

    # catboost_regressor_pipeline = CatBoostRegressorPipeline()
    # catboost_regressor_pipeline.main()

    #----------------------------- AGGREGATE RESULTS -----------------------------#

    results_aggregator = ResultsAggregator()
    results_aggregator.main()
