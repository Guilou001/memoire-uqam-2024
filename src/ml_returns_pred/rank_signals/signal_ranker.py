from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


class SignalsRankerAbstract(ABC):
    """
    Abstract base class for different signal ranking strategies.
    """

    @abstractmethod
    def rank_signals(self, signals: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


class SimpleSignalsRanker(SignalsRankerAbstract):
    """
    This class implements a simple ranking strategy for signals in a DataFrame.
    """
    def __init__(self):
        pass

    def rank_signals(self, signals: pd.DataFrame, ascending: bool = False,
                     method: Literal["average", "min", "max", "first", "dense"] = 'first') -> pd.DataFrame:
        return signals.rank(axis=1, ascending=ascending, method=method)


class RankerFactory:
    """
    Factory class to create instances of signal ranking strategies.
    """
    strategies = {
        "simple": SimpleSignalsRanker,
    }

    @classmethod
    def select_ranker(cls, ranking_strategy: str) -> SignalsRankerAbstract:
        """
        Selects and initializes the specified ranking strategy with provided arguments.

        Parameters:
        ----------
        strategy_type : str
            The type of the ranking strategy to create.
        **kwargs
            Additional keyword arguments to pass to the strategy initializer.

        Returns:
        -------
        SignalsRankerFactory
            An instance of the specified ranking strategy.

        Raises:
        ------
        ValueError
            If the strategy type is not recognized.
        """
        strategy_class = cls.strategies.get(ranking_strategy)
        if strategy_class:
            return strategy_class()
        else:
            valid_types = ", ".join(cls.strategies.keys())
            raise ValueError(f"Unknown ranker type: '{ranking_strategy}'. Valid types are: {valid_types}.")


if __name__ == '__main__':
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.prediction_pipeline.prediction_pipeline import PredictionPipeline
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
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



