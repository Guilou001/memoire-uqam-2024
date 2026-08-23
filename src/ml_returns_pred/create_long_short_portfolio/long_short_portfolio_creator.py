import numpy as np
import pandas as pd

from ml_returns_pred.rank_signals.signal_ranker import RankerFactory

pd.set_option('future.no_silent_downcasting', True)


class LongShortPortfolioCreatorFromRanking:
    """
    This class creates long and short portfolios from signals, based on specified thresholds on ranked signals.
    It allows for dynamic selection of ranking strategies and portfolio construction methods.
    """

    def __init__(self, signals: pd.DataFrame) -> None:
        """
        Initializes the LongShortPortfolioCreatorFromRanking with signal data.

        Parameters:
        - signals: A DataFrame containing the signals for all assets.
        """
        self.signals = self._validate_and_clean_signals(signals)

    @staticmethod
    def _validate_and_clean_signals(signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and cleans the input signals DataFrame by removing rows where all values are NaN.

        Parameters:
        - signals: A DataFrame containing the signals for all assets.

        Returns:
        - A cleaned DataFrame with rows containing all NaN values removed.
        """
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("The signals must be provided as a pandas DataFrame.")
        return signals.dropna(axis=0, how='all')

    def create_long_short_portfolio(self, ranking_strategy: str = "simple",
                                    ascending: bool = True,
                                    method: str = 'first',
                                    selection_method: str = "percentile",
                                    percentile_threshold: float = 0.1,
                                    fix_threshold: int = 20,
                                    keep_signal_value: bool = False,
                                    use_ranking_as_signals: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates and returns long and short portfolios based on the selection method and thresholds using the specified ranking strategy.

        Parameters:
        - ranking_strategy: The strategy used to rank signals.
        - ascending: Whether to sort rankings in ascending order.
        - method: The method to use for breaking ties in rankings.
        - selection_method: The method to select long/short positions ('percentile' or 'fix').
        - percentile_threshold: The percentile threshold for selecting long/short positions.
        - fix_threshold: The fixed threshold for selecting long/short positions.
        - keep_signal_value: If True, keeps the original signal values in the long/short signals.
        - use_ranking_as_signals: If True, uses the ranking as the final signals for long/short positions.

        Returns:
        - A tuple containing the DataFrames for long_signals and short_signals.
        """
        ranked_signals = self._rank_signals(ranking_strategy, ascending, method)

        if use_ranking_as_signals:
            return ranked_signals, ranked_signals

        long_signals, short_signals = self._select_signals(ranked_signals, selection_method, percentile_threshold, fix_threshold)

        if keep_signal_value:
            long_signals = self.signals.where(long_signals.notna(), np.nan)
            short_signals = self.signals.where(short_signals.notna(), np.nan)

        return long_signals, short_signals

    def _rank_signals(self, ranking_strategy: str, ascending: bool, method: str) -> pd.DataFrame:
        """
        Ranks the signals using the specified ranking strategy.

        Parameters:
        - ranking_strategy: The strategy used to rank signals.
        - ascending: Whether to sort rankings in ascending order.
        - method: The method to use for breaking ties in rankings.

        Returns:
        - A DataFrame of ranked signals.
        """
        ranker_factory = RankerFactory()
        ranker = ranker_factory.select_ranker(ranking_strategy)
        return ranker.rank_signals(self.signals, ascending=ascending, method=method)

    def _select_signals(self, ranked_signals: pd.DataFrame, selection_method: str,
                        percentile_threshold: float, fix_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Selects long and short signals based on the selection method.

        Parameters:
        - ranked_signals: A DataFrame of ranked signals.
        - selection_method: The method to select long/short positions ('percentile' or 'fix').
        - percentile_threshold: The percentile threshold for selecting long/short positions.
        - fix_threshold: The fixed threshold for selecting long/short positions.

        Returns:
        - A tuple containing the DataFrames for long_signals and short_signals.
        """
        if selection_method == "percentile":
            return self._apply_percentile_method(ranked_signals, percentile_threshold)
        elif selection_method == "fix":
            return self._apply_fix_method(ranked_signals, fix_threshold)
        else:
            raise ValueError(f"Invalid selection method: {selection_method}. Valid methods are 'percentile' or 'fix'.")

    @staticmethod
    def _apply_percentile_method(ranked_signals: pd.DataFrame, percentile_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the percentile method to select long and short signals.

        Parameters:
        - ranked_signals: A DataFrame of ranked signals.
        - percentile_threshold: The percentile threshold for selecting long/short positions.

        Returns:
        - A tuple containing the DataFrames for long_signals and short_signals.
        """
        long_signals = ranked_signals <= ranked_signals.quantile(percentile_threshold, axis=1)
        short_signals = ranked_signals >= ranked_signals.quantile(1 - percentile_threshold, axis=1)
        return long_signals.astype(float).replace(False, np.nan), short_signals.astype(float).replace(False, np.nan)

    @staticmethod
    def _apply_fix_method(ranked_signals: pd.DataFrame, fix_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the fixed threshold method to select long and short signals.

        Parameters:
        - ranked_signals: A DataFrame of ranked signals.
        - fix_threshold: The fixed threshold for selecting long/short positions.

        Returns:
        - A tuple containing the DataFrames for long_signals and short_signals.
        """
        long_signals = ranked_signals <= fix_threshold
        short_signals = ranked_signals.apply(lambda row: row >= row.nlargest(fix_threshold).min(), axis=1)
        return long_signals.astype(float).replace(False, np.nan), short_signals.astype(float).replace(False, np.nan)


class LongShortPortfolioCreatorFromSignals:
    """
    This class is designed to construct long and short portfolios based on signal data provided as input.
    A positive signal suggests a long position, whereas a negative signal suggests a short position. The class
    allows for an optional limitation on the number of tickers per date to be included in either portfolio,
    randomly selecting tickers if this limit is exceeded.

    Attributes
    ----------
    signals : pd.DataFrame
        A DataFrame containing signals for various tickers across different dates. Each entry in the DataFrame
        should represent the signal for a specific ticker on a specific date, where positive values indicate
        potential long positions and negative values suggest potential short positions.
    maximum_tickers_per_date : int, optional
        The maximum number of tickers to be included in each of the long and short portfolios for any given date.
        If the number of available tickers exceeds this limit, a random selection is made to meet the specified
        maximum. If None, all tickers with valid signals are included without limitation.

    Methods
    -------
    create_long_short_portfolio() -> tuple[pd.DataFrame, pd.DataFrame]
        Processes the input signals to create two DataFrames, one for long positions and one for short positions,
        based on the sign of the signals. If maximum_tickers_per_date is specified, limits the number of tickers
        included in each portfolio per date through random selection.

    _select_random_tickers(signals: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame
        A helper method used to randomly select a subset of tickers for inclusion in the portfolio on each date,
        adhering to the maximum_tickers_per_date constraint. The random seed ensures reproducibility of the
        selection process.

    Example
    -------
    signals = pd.DataFrame({'Ticker1': [1, -1, 0], 'Ticker2': [-1, 1, 1]}, index=['2021-01-01', '2021-01-02', '2021-01-03'])
    portfolio_creator = LongShortPortfolioCreatorFromSignals(signals, maximum_tickers_per_date=1)
    long_portfolio, short_portfolio = portfolio_creator.create_long_short_portfolio()

    This example demonstrates initializing the class with a DataFrame of signals indicating potential long and short
    positions. It then creates long and short portfolios with a constraint of including no more than one ticker per
    date in each portfolio.
    """
    def __init__(self, signals: pd.DataFrame, maximum_tickers_per_date: int = None):
        self.signals = signals.dropna(axis=0, how='all')
        self.maximum_tickers_per_date = maximum_tickers_per_date

    def create_long_short_portfolio(self, keep_signal_value: bool = False, transform_continuous_to_binary: bool = False,
                                    random_seed: int = 42,
                                    transform_binary_classification_to_rank: bool = True) -> (
            tuple)[pd.DataFrame, pd.DataFrame]:
        np.random.seed(random_seed)

        if transform_continuous_to_binary and transform_binary_classification_to_rank:
            long_signals = self.signals > 0.0
            long_signals = long_signals.replace({False: 2.0, True: 1.0})
            short_signals = long_signals

        elif keep_signal_value:
            long_signals = self.signals[self.signals > 0.0]
            short_signals = self.signals[self.signals <= 0.0]

        elif transform_binary_classification_to_rank:
            long_signals = self.signals.replace({0: 2, 1: 1})
            short_signals = long_signals

        else:
            # Create masks for long and short signals, replacing False with np.nan
            long_signals = pd.DataFrame(self.signals > 0.0)
            short_signals = pd.DataFrame(self.signals <= 0.0)
            long_signals = long_signals.replace(False, np.nan)
            short_signals = short_signals.replace(False, np.nan)

        if self.maximum_tickers_per_date is not None:
            long_signals = self._select_random_tickers(long_signals)
            short_signals = self._select_random_tickers(short_signals)

        # If not keeping signal values, ensure the output is boolean for True signals and NaN otherwise
        if not keep_signal_value and not transform_binary_classification_to_rank:
            long_signals = long_signals.where(long_signals, np.nan)
            short_signals = short_signals.where(short_signals, np.nan)

        return long_signals, short_signals

    def _select_random_tickers(self, signals: pd.DataFrame) -> pd.DataFrame:
        selected_signals = pd.DataFrame(index=signals.index, columns=signals.columns)

        for date in signals.index:
            available_tickers = signals.loc[date].dropna().index  # Drop NaN to only consider available tickers

            # Select tickers randomly if exceeding the maximum limit, else select all
            if len(available_tickers) > self.maximum_tickers_per_date:
                selected_tickers = np.random.choice(available_tickers, self.maximum_tickers_per_date, replace=False)
            else:
                selected_tickers = available_tickers

            # Update selected signals accordingly
            selected_signals.loc[date, selected_tickers] = signals.loc[date, selected_tickers]

        return selected_signals.where(signals.notna(), np.nan)  # Ensure non-selected are NaN


if __name__ == '__main__':
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.prediction_pipeline.prediction_pipeline import PredictionPipeline
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.rank_signals.signal_ranker import RankerFactory
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.resample_data.data_resampler import DataResampler

    dr = DataReader()
    relative_data_path = "../../data/raw_data/canadian_stocks_data_10.csv"
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
    endogenous_pipeline_dict = {
        "detrend": {},
        "robust_scaler": {"with_centering": True},
    }

    exogenous_pipeline_dict = {
        "robust_scaler": {"with_centering": True},
    }

    # Paramètres du forecaster
    forecaster_name = "lasso"
    forecaster_params = {"alpha": 0.1}
    make_reduction_params = {"strategy": "recursive", "window_length": 12}

    # ------------------------------------- #

    # Initialisation et utilisation de la classe PredictionPipelineSktime
    pp = PredictionPipeline(endogenous_data=returns, exogenous_data=macro_data_resampled)
    pp.fit_endogenous_and_exogenous_data(
        endogenous_pipeline_dict=endogenous_pipeline_dict,
        exogenous_pipeline_dict=exogenous_pipeline_dict,
        forecaster_name=forecaster_name,
        forecaster_params=forecaster_params,
        make_reduction_params=make_reduction_params
    )
    predictions = pp.predict()

    print(predictions)

    rf = RankerFactory()
    ranker = rf.select_ranker(ranking_strategy='simple')
    ranked_signals = ranker.rank_signals(signals=predictions, ascending=False, method='first')

    print(ranked_signals)

    ranking_strategy = "simple"
    ascending = True
    method = 'first'
    selection_method = "fix"
    percentile_threshold = 0.1
    fix_threshold = 3
    keep_signal_value = False

    lsc = LongShortPortfolioCreatorFromRanking(
        signals=ranked_signals,
    )

    long_signals, short_signals = lsc.create_long_short_portfolio(
        ranking_strategy=ranking_strategy,
        ascending=ascending,
        method=method,
        selection_method=selection_method,
        percentile_threshold=percentile_threshold,
        fix_threshold=fix_threshold,
        keep_signal_value=keep_signal_value
    )

    print(long_signals)
    print(short_signals)

