
import numpy as np
import pandas as pd
from numba import njit, prange


@njit
def calculate_drifted_weights_numba(weights_aligned: np.ndarray, daily_returns: np.ndarray, is_long: bool,
                                    rebalance_indices: np.ndarray) -> np.ndarray:
    n_dates, n_assets = daily_returns.shape
    drifted_weights = np.zeros((n_dates, n_assets))

    current_weights = weights_aligned[0]
    drifted_weights[0] = current_weights

    for i in prange(1, n_dates):
        if i in rebalance_indices:
            current_weights = weights_aligned[i]
        else:
            return_factor = (1 + daily_returns[i]) if is_long else (1 - daily_returns[i])
            daily_change = current_weights * return_factor
            daily_change[np.isnan(daily_returns[i])] = 0

            if np.sum(current_weights != 0) > 1:
                current_weights = daily_change / np.abs(daily_change).sum()
            else:
                current_weights = daily_change

        drifted_weights[i] = current_weights

    return drifted_weights


@njit
def calculate_transaction_costs_numba(drifted_weights_long: np.ndarray, drifted_weights_short: np.ndarray,
                                      is_long_only: bool, transaction_fee: float,
                                      rebalance_indices: np.ndarray) -> np.ndarray:
    n_dates, n_assets = drifted_weights_long.shape
    transaction_costs = np.zeros(n_dates)

    for i in prange(1, n_dates):
        if i in rebalance_indices:
            current_weights_long = drifted_weights_long[i]
            prev_weights_long = drifted_weights_long[i - 1]
            weight_changes_long = np.abs(current_weights_long - prev_weights_long)
            transaction_costs_long = weight_changes_long.sum() * transaction_fee

            total_transaction_costs = transaction_costs_long

            if not is_long_only:
                current_weights_short = drifted_weights_short[i]
                prev_weights_short = drifted_weights_short[i - 1]
                weight_changes_short = np.abs(current_weights_short - prev_weights_short)
                transaction_costs_short = weight_changes_short.sum() * transaction_fee
                total_transaction_costs += transaction_costs_short

            transaction_costs[i] = total_transaction_costs

    return transaction_costs


class StrategyReturnsCalculatorOptimized:
    def __init__(self, long_weights: pd.DataFrame, short_weights: pd.DataFrame | None,
                 prices_data_preprocessed: pd.DataFrame,
                 implementation_days_delta: int = 0, is_long_only: bool = True,
                 transaction_fee: float = 0.001) -> None:
        """
        Initialize the StrategyReturnsCalculator.

        Parameters:
        long_weights (pd.DataFrame): DataFrame of long position weights.
        short_weights (Optional[pd.DataFrame]): DataFrame of short position weights (if applicable).
        prices_data_preprocessed (pd.DataFrame): Preprocessed DataFrame of prices.
        implementation_days_delta (int): Days to shift weights for implementation.
        is_long_only (bool): Flag indicating if the strategy is long-only.
        transaction_fee (float): Transaction fee rate.
        """
        self.long_weights = self._verify_datetime_index(df=long_weights)
        self.short_weights = self._verify_datetime_index(df=short_weights) if not is_long_only else None
        self.prices_data_preprocessed = self._verify_datetime_index(df=prices_data_preprocessed)
        self.daily_returns = self.prices_data_preprocessed.pct_change().iloc[1:]
        self.is_long_only = is_long_only
        self.transaction_fee = transaction_fee
        self.implementation_days_delta = implementation_days_delta
        self.drifted_weights_long = None
        self.drifted_weights_short = None
        self.rebalance_indices = None

    @staticmethod
    def _verify_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame index is a DatetimeIndex.

        Parameters:
        df (pd.DataFrame): DataFrame to check.

        Returns:
        pd.DataFrame: DataFrame with DatetimeIndex.
        """
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp(how='end').normalize()
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def calculate_drifted_weights(self) -> None:
        """
        Calculate drifted weights for long and short positions.
        """
        self.drifted_weights_long = self._get_drifted_weights(weights=self.long_weights, is_long=True)
        if not self.is_long_only:
            self.drifted_weights_short = self._get_drifted_weights(weights=self.short_weights, is_long=False)

    def concat_daily_returns_with_risk_free_rates(self, risk_free_rates: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate daily returns with risk-free rates.

        Parameters:
        risk_free_rates (pd.DataFrame): DataFrame containing the risk-free rates.

        Returns:
        pd.DataFrame: DataFrame with daily returns and reindexed risk-free rates.
        """
        risk_free_rates_shifted = risk_free_rates.shift(1)
        risk_free_rates_reindexed = risk_free_rates_shifted.reindex(self.daily_returns.index).fillna(0)
        self.daily_returns = pd.concat([self.daily_returns, risk_free_rates_reindexed], axis=1).dropna(axis=0, how='any')
        return self.daily_returns

    def _get_drifted_weights(self, weights: pd.DataFrame, is_long: bool) -> pd.DataFrame:
        """
        Get drifted weights adjusted for implementation delay.

        Parameters:
        weights (pd.DataFrame): DataFrame of weights to drift.
        is_long (bool): Boolean flag indicating if the weights are for long positions.

        Returns:
        pd.DataFrame: DataFrame of drifted weights.
        """
        weights_aligned, daily_returns = weights.align(self.daily_returns, axis=1, join='inner')
        weights_aligned.index += pd.Timedelta(days=self.implementation_days_delta)

        aligned_start_date = max(weights_aligned.index.min(), daily_returns.index.min())
        daily_returns = daily_returns.loc[aligned_start_date:]
        weights_aligned = weights_aligned.loc[aligned_start_date:]

        weights_aligned_reindexed = weights_aligned.reindex(daily_returns.index, method='ffill')

        self.rebalance_indices = np.searchsorted(daily_returns.index, self.long_weights.index)

        weights_aligned_np = weights_aligned_reindexed.to_numpy(dtype=np.float64)
        daily_returns_np = daily_returns.to_numpy(dtype=np.float64)

        drifted_weights_np = calculate_drifted_weights_numba(weights_aligned=weights_aligned_np,
                                                             daily_returns=daily_returns_np,
                                                             is_long=is_long,
                                                             rebalance_indices=self.rebalance_indices)
        drifted_weights = pd.DataFrame(data=drifted_weights_np, index=daily_returns.index, columns=weights_aligned.columns)

        return drifted_weights

    def compute_strategy_returns(self) -> pd.DataFrame:
        """
        Compute the overall portfolio returns.

        Returns:
        pd.DataFrame: DataFrame of portfolio returns.
        """
        start_date = self.long_weights.index[0]
        daily_returns = self.daily_returns.loc[start_date:]

        strategy_returns_long = (self.drifted_weights_long.shift(1) * daily_returns).sum(axis=1)
        strategy_returns = strategy_returns_long
        if not self.is_long_only:
            strategy_returns_short = (self.drifted_weights_short.shift(1) * daily_returns).sum(axis=1)
            strategy_returns += strategy_returns_short

        portfolio_returns = pd.DataFrame(data=strategy_returns, columns=["Portfolio_Returns"])
        transaction_costs = self._calculate_transaction_costs_at_rebalance_dates()
        portfolio_returns["Portfolio_Returns"] -= transaction_costs

        return portfolio_returns

    def _calculate_transaction_costs_at_rebalance_dates(self) -> pd.Series:
        """
        Calculate transaction costs at rebalance dates.

        Returns:
        pd.Series: Series of transaction costs.
        """
        drifted_weights_long_np = self.drifted_weights_long.to_numpy(dtype=np.float64)
        drifted_weights_short_np = self.drifted_weights_short.to_numpy(dtype=np.float64) if not self.is_long_only \
            else np.zeros_like(drifted_weights_long_np)

        transaction_costs_np = calculate_transaction_costs_numba(drifted_weights_long=drifted_weights_long_np,
                                                                 drifted_weights_short=drifted_weights_short_np,
                                                                 is_long_only=self.is_long_only,
                                                                 transaction_fee=self.transaction_fee,
                                                                 rebalance_indices=self.rebalance_indices)

        transaction_costs = pd.Series(data=transaction_costs_np, index=self.drifted_weights_long.index)
        return transaction_costs


if __name__ == "__main__":
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.create_long_short_portfolio.long_short_portfolio_creator import (
        LongShortPortfolioCreatorFromRanking,
    )
    from ml_returns_pred.prediction_pipeline.prediction_pipeline import PredictionPipeline
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.rank_signals.signal_ranker import RankerFactory
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.resample_data.data_resampler import DataResampler
    from ml_returns_pred.weighting.weighting import WeightingStrategyFactory

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

    lsc = LongShortPortfolioCreatorFromRanking(
        signals=ranked_signals,
    )

    long_signals, short_signals = lsc.create_long_short_portfolio(
        ranking_strategy="simple",
        ascending=True,
        method='first',
        selection_method="fix",
        percentile_threshold=0.1,
        fix_threshold=3,
        keep_signal_value=False
    )

    print(long_signals)
    print(short_signals)

    weighting_strategy = "equal_weighting"

    wsf = WeightingStrategyFactory()
    weighting_strategy_instance = wsf.select_weighting_strategy(weighting_strategy)

    long_weights, short_weights = weighting_strategy_instance.compute_weights(
        long_signals=long_signals,
        short_signals=short_signals
    )

    print(long_weights)
    print(short_weights)

    strategy_returns_calculator = StrategyReturnsCalculatorOptimized(
        long_weights=long_weights,
        short_weights=short_weights,
        prices_data_preprocessed=prices_data_preprocessed,
        implementation_days_delta=0,
        is_long_only=True,
        transaction_fee=0.001
    )

    strategy_returns_calculator.calculate_drifted_weights()
    portfolio_returns = strategy_returns_calculator.compute_strategy_returns()
    print(portfolio_returns)
