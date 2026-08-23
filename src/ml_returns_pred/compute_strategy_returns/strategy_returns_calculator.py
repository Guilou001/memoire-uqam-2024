import numpy as np
import pandas as pd
from tqdm import tqdm


class StrategyReturnsCalculator:
    LONG_SHORT_MODES = ("as_published", "corrected")

    def __init__(self, long_weights: pd.DataFrame, short_weights: pd.DataFrame,
                 prices_data_preprocessed: pd.DataFrame,
                 implementation_days_delta: int = 0, is_long_only: bool = True,
                 transaction_fee: float = 0.001, long_short_mode: str = "as_published") -> None:
        """
        Initialize the StrategyReturnsCalculator.

        Parameters:
        long_weights (pd.DataFrame): DataFrame of long position weights.
        short_weights (pd.DataFrame): DataFrame of short position weights (if applicable).
        prices_data_preprocessed (pd.DataFrame): Preprocessed DataFrame of prices.
        implementation_days_delta (int): Days to shift weights for implementation.
        is_long_only (bool): Flag indicating if the strategy is long-only.
        transaction_fee (float): Transaction fee rate.
        long_short_mode (str): ``"as_published"`` reproduit le code du mémoire (2024) : les poids
            de la jambe « short » sont positifs, ses rendements sont ADDITIONNÉS à ceux de la jambe
            longue, et les deux jambes dérivent avec le facteur (1 - r). Le portefeuille est donc
            long sur les deux jambes (exposition brute 200 %). ``"corrected"`` calcule un vrai
            long-short : rendement = jambe longue - jambe courte, dérive (1 + r) pour les deux jambes.
        """
        if long_short_mode not in self.LONG_SHORT_MODES:
            raise ValueError(f"long_short_mode doit être dans {self.LONG_SHORT_MODES}, reçu {long_short_mode!r}")
        self.long_weights = self._verify_datetime_index(long_weights)
        self.short_weights = self._verify_datetime_index(short_weights) if not is_long_only else None
        self.prices_data_preprocessed = self._verify_datetime_index(prices_data_preprocessed)
        self.daily_returns = self.prices_data_preprocessed.pct_change().iloc[1:]
        self.is_long_only = is_long_only
        self.transaction_fee = transaction_fee
        self.implementation_days_delta = implementation_days_delta
        self.long_short_mode = long_short_mode
        self.drifted_weights_long = None
        self.drifted_weights_short = None

    @staticmethod
    def _verify_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame index is a DatetimeIndex.

        Parameters:
        df (pd.DataFrame): DataFrame to check.

        Returns:
        pd.DataFrame: DataFrame with DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def calculate_drifted_weights(self) -> None:
        """
        Calculate drifted weights for long and short positions.
        """
        self.drifted_weights_long = self._get_drifted_weights(self.long_weights)
        if not self.is_long_only:
            self.drifted_weights_short = self._get_drifted_weights(self.short_weights)

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

    def _get_drifted_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Get drifted weights adjusted for implementation delay.

        Parameters:
        weights (pd.DataFrame): DataFrame of weights to drift.

        Returns:
        pd.DataFrame: DataFrame of drifted weights.
        """
        weights_aligned, daily_returns = weights.align(self.daily_returns, axis=1, join='inner')
        weights_aligned.index += pd.Timedelta(days=self.implementation_days_delta)
        weights_aligned.fillna(0, inplace=True)

        aligned_start_date = max(weights_aligned.index.min(), daily_returns.index.min())
        daily_returns = daily_returns.loc[aligned_start_date:]
        weights_aligned = weights_aligned.loc[aligned_start_date:]

        drifted_weights = pd.DataFrame(index=daily_returns.index, columns=weights_aligned.columns)
        current_weights = weights_aligned.iloc[0]
        drifted_weights.iloc[0] = current_weights

        for date in tqdm(daily_returns.index[1:], desc="Calculating drifted weights"):
            if date in weights_aligned.index:
                current_weights = weights_aligned.loc[date]
            else:
                if self.long_short_mode == "corrected" or self.is_long_only:
                    # dérive de la valeur de chaque position : (1 + r) pour les deux jambes
                    return_factor = (1 + daily_returns.loc[date])
                else:
                    # code du mémoire (2024) : facteur (1 - r) appliqué aux deux jambes dès que la
                    # stratégie n'est pas long seul ; conservé tel quel pour la reproductibilité
                    return_factor = (1 - daily_returns.loc[date])
                daily_change = current_weights * return_factor
                daily_change[daily_returns.loc[date].isna()] = 0

                if (current_weights != 0).sum() > 1:
                    current_weights = daily_change / abs(daily_change).sum()
                else:
                    current_weights = daily_change

            drifted_weights.loc[date] = current_weights

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
            if self.long_short_mode == "corrected":
                # vrai long-short : on gagne quand les titres vendus à découvert baissent
                strategy_returns = strategy_returns - strategy_returns_short
            else:
                # code du mémoire (2024) : la jambe « short » est additionnée (portefeuille long des deux côtés)
                strategy_returns = strategy_returns + strategy_returns_short

        portfolio_returns = pd.DataFrame(strategy_returns, columns=["Portfolio_Returns"])
        transaction_costs = self._calculate_transaction_costs_at_rebalance_dates()
        portfolio_returns["Portfolio_Returns"] -= transaction_costs

        portfolio_returns["Portfolio_Returns"] = pd.to_numeric(portfolio_returns["Portfolio_Returns"], errors='coerce')

        return portfolio_returns

    def _calculate_transaction_costs_at_rebalance_dates(self) -> pd.Series:
        """
        Calculate transaction costs at rebalance dates.

        Returns:
        pd.Series: Series of transaction costs.
        """
        transaction_costs = pd.Series(0.0, index=self.prices_data_preprocessed.index)

        for date_pos, date in enumerate(tqdm(self.long_weights.index, desc="Calculating transaction costs")):
            prev_date_pos = date_pos - 1

            if prev_date_pos < 0:
                continue

            current_weights_long = self.drifted_weights_long.iloc[date_pos]
            prev_weights_long = self.drifted_weights_long.iloc[prev_date_pos]
            weight_changes_long = np.abs(current_weights_long - prev_weights_long)
            transaction_costs_long = weight_changes_long.sum() * self.transaction_fee

            if not self.is_long_only:
                current_weights_short = self.drifted_weights_short.iloc[date_pos]
                prev_weights_short = self.drifted_weights_short.iloc[prev_date_pos]
                weight_changes_short = np.abs(current_weights_short - prev_weights_short)
                transaction_costs_short = weight_changes_short.sum() * self.transaction_fee
                total_transaction_costs = transaction_costs_long + transaction_costs_short
            else:
                total_transaction_costs = transaction_costs_long

            transaction_costs[date] = total_transaction_costs

        return transaction_costs
