from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class WeightingStrategyAbstract(ABC):
    """
    Abstract base class for different weighting strategies in a portfolio.
    """

    @abstractmethod
    def compute_weights(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Abstract method to compute weights for long and short positions.
        Must be implemented by concrete subclasses.

        Parameters
        ----------
        long_signals : DataFrame
            A DataFrame with boolean values indicating long positions.
        short_signals : DataFrame
            A DataFrame with boolean values indicating short positions.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames: long_weights and short_weights.
        """
        pass


class RankBasedWeightingStrategy(WeightingStrategyAbstract):
    """
    This class implements a weighting strategy based on the ranking of signals.
    The weights are assigned such that tickers with the best rankings are over-weighted in the long portfolio,
    and those with the worst rankings are over-weighted in the short portfolio.
    """

    def __init__(self, method: str = 'linear'):
        """
        Initialize the RankBasedWeightingStrategy.

        Parameters
        ----------
        method : str
            The method to use for weighting ('linear', 'exponential', 'inverse', 'random', 'equal').
        """
        self.method = method

    def compute_weights(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame, fix_threshold: int, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute weights for long and short positions based on rankings.

        Parameters
        ----------
        long_signals : DataFrame
            A DataFrame indicating long positions. Contains rankings.
        short_signals : DataFrame
            A DataFrame indicating short positions. Contains rankings.
        fix_threshold : int
            The number of top-ranked tickers to allocate weights to.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames: long_weights and short_weights, with weights based on rankings.
        """
        long_weights = self._compute_weights_from_ranks(long_signals, fix_threshold=fix_threshold)
        short_weights = self._compute_weights_from_ranks(short_signals, fix_threshold=fix_threshold, reverse=True)

        return long_weights, short_weights

    def _compute_weights_from_ranks(self, ranks: pd.DataFrame, fix_threshold: int, reverse: bool = False) -> pd.DataFrame:
        """
        Compute weights from ranks. Higher ranks (better performance) get more weight in long portfolio,
        and lower ranks (worse performance) get more weight in short portfolio.

        Parameters
        ----------
        ranks : DataFrame
            A DataFrame containing the ranks of the tickers.
        fix_threshold : int
            The number of top-ranked (or bottom-ranked if reverse is True) tickers to allocate weights to.
        reverse : bool
            If True, reverse the ranking weights for the short portfolio.

        Returns
        -------
        DataFrame
            A DataFrame with computed weights.
        """
        np.random.seed(42)  # Set seed for reproducibility

        if reverse:
            mask = ranks >= (ranks.max().max() - fix_threshold + 1)
        else:
            mask = ranks <= fix_threshold

        if self.method == 'linear':
            weights = mask * ranks
            weights = ranks.shape[1] - weights + 1
        elif self.method == 'exponential':
            weights = mask * (2 ** (ranks.shape[1] - ranks) if not reverse else 2 ** (ranks - 1))
        elif self.method == 'inverse':
            weights = mask * (1 / ranks)
        elif self.method == 'random':
            weights = mask * pd.DataFrame(np.random.rand(*ranks.shape), index=ranks.index, columns=ranks.columns)
        elif self.method == 'equal':
            weights = mask.astype(float)
        else:
            raise ValueError("Invalid method. Choose 'linear', 'exponential', 'inverse', 'random', or 'equal'.")

        # Set weights to zero for any rows where the mask is entirely False
        weights[mask.sum(axis=1) == 0] = 0

        # Normalize the weights to sum to 1 for each date
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)

        return weights


class EqualWeightingStrategy(WeightingStrategyAbstract):
    """
    This class implements a simplified and efficient equal weighting strategy for a portfolio,
    assigning equal weights to all non-NaN signals for each date.
    """

    def compute_weights(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame, **kwargs) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Compute equal weights for long and short positions based on non-NaN signals.

        Parameters
        ----------
        long_signals : DataFrame
            A DataFrame indicating long positions. Non-NaN values are considered active signals.
        short_signals : DataFrame
            A DataFrame indicating short positions. Non-NaN values are considered active signals.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames: long_weights and short_weights, with equal weights for non-NaN signals.
        """
        long_weights = self._compute_equal_weights(long_signals)
        short_weights = self._compute_equal_weights(short_signals)

        return long_weights, short_weights

    @staticmethod
    def _compute_equal_weights(signals: pd.DataFrame) -> pd.DataFrame:
        """
        Computes equal weights for each non-NaN signal in the DataFrame.

        Parameters
        ----------
        signals : DataFrame
            A DataFrame with signals for positions. Non-NaN values are considered active signals.

        Returns
        -------
        DataFrame
            A DataFrame with equal weights assigned to each non-NaN signal.
        """
        # Count non-NaN values in each row to determine the number of active signals
        active_signals_count = signals.notna().sum(axis=1)

        # Calculate equal weight for each active signal by dividing 1 by the number of active signals
        weights = signals.notna().div(active_signals_count, axis=0)

        # Replace NaNs with 0 for rows/columns where there are no active signals or the signal itself is NaN
        weights.fillna(0, inplace=True)

        return weights


class WeightingFromPureSignalsWith2Assets(WeightingStrategyAbstract):

    def compute_weights(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame, **kwargs) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute weights for long and short positions based on signal values.

        Parameters
        ----------
        long_signals : DataFrame
            A DataFrame containing the signal values for long positions.
        short_signals : DataFrame
            A DataFrame containing the signal values for short positions.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames: long_weights and short_weights, with weights proportional to the signals' magnitudes.
        """
        long_weights = long_signals.copy()
        short_weights = short_signals.copy()

        long_weights['Risk Free Asset'] = 1 - long_signals[long_signals.columns[0]]
        short_weights['Risk Free Asset'] = 0

        for date, row in long_signals.iterrows():
            if row.isna().iloc[0]:
                long_weights.loc[date, 'Risk Free Asset'] = 1 - short_signals.loc[date, short_signals.columns[0]]

        return long_weights.fillna(0), short_weights.fillna(0)


class WeightingFromNormalizedSignals(WeightingStrategyAbstract):
    """
    This class implements a weighting strategy based on the magnitude of signals for a portfolio.
    The weights are proportional to the signal values, allowing for a dynamic allocation based on the strength of the signals.
    """

    def compute_weights(self, long_signals: pd.DataFrame, short_signals: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute weights for long and short positions based on signal values.

        Parameters
        ----------
        long_signals : DataFrame
            A DataFrame containing the signal values for long positions.
        short_signals : DataFrame
            A DataFrame containing the signal values for short positions.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames: long_weights and short_weights, with weights proportional to the signals' magnitudes.
        """
        long_weights = self._compute_signal_based_weights(long_signals)
        short_weights = self._compute_signal_based_weights(short_signals)

        return long_weights, short_weights

    @staticmethod
    def _compute_signal_based_weights(signals: pd.DataFrame) -> pd.DataFrame:
        """
        Computes weights based on the magnitude of signals. The function normalizes the signal values in each row
        to sum to 1 (or -1 for short positions), allowing the portfolio to allocate more weight to stronger signals.

        Parameters
        ----------
        signals : DataFrame
            A DataFrame with signal values for either long or short positions.

        Returns
        -------
        DataFrame
            A DataFrame with weights proportional to the signal magnitudes.
        """
        # For long positions, normalize signal values to sum to 1
        # For short positions, you might want to invert the signals before normalization if they're not already negative
        # This example assumes the signals for short positions are already prepared for weighting
        absolute_signals = signals.abs()
        weights = absolute_signals.div(absolute_signals.sum(axis=1), axis=0)
        weights.fillna(0, inplace=True)  # Replace NaNs with 0 for days with no signals

        return weights


class WeightingStrategyFactory:
    """
    Factory class to create instances of weighting strategies.
    """

    strategies = {
        "equal_weighting": EqualWeightingStrategy,
        "weighting_from_normalized_signals": WeightingFromNormalizedSignals,
        "weighting_from_pure_signals_with_2_assets": WeightingFromPureSignalsWith2Assets,
        "rank_based_weighting": RankBasedWeightingStrategy  # Ajout de la nouvelle stratégie
    }

    @classmethod
    def select_weighting_strategy(cls, strategy_type: str, **kwargs) -> WeightingStrategyAbstract:
        """
        Selects the specified weighting strategy.

        Parameters
        ----------
        strategy_type : str
            The type of the weighting strategy to create.

        Returns
        -------
        WeightingStrategyAbstract
            An instance of the specified weighting strategy.

        Raises
        ------
        ValueError
            If the strategy type is not recognized.
        """
        strategy_class = cls.strategies.get(strategy_type)
        if strategy_class:
            return strategy_class(**kwargs)
        else:
            valid_types = ", ".join(cls.strategies.keys())
            raise ValueError(f"Unknown strategy type: {strategy_type}. "
                             f"Valid types are: {valid_types}.")


if __name__ == '__main__':
    # Création d'un DataFrame de test pour les classements de signaux
    data = {
        'Ticker1': [1, 2, 3, 4],
        'Ticker2': [2, 1, 4, 3],
        'Ticker3': [3, 4, 2, 1],
        'Ticker4': [4, 3, 1, 2]
    }
    index = pd.date_range(start='2023-01-01', periods=4, freq='ME')
    ranking_signals = pd.DataFrame(data, index=index)

    print("Ranking Signals:\n", ranking_signals)

    # Sélection de la stratégie de pondération basée sur le classement
    weighting_strategy = "rank_based_weighting"
    method = "exponential"  # Choisissez entre 'linear', 'exponential', 'inverse'

    wsf = WeightingStrategyFactory()
    weighting_strategy_instance = wsf.select_weighting_strategy(weighting_strategy, method=method)

    long_weights, short_weights = weighting_strategy_instance.compute_weights(
        long_signals=ranking_signals,
        short_signals=ranking_signals  # Pour le test, on utilise les mêmes signaux pour les deux portefeuilles
    )

    print("\nLong Weights:\n", long_weights)
    print("\nShort Weights:\n", short_weights)