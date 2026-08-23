from datetime import datetime
from typing import Literal, Union

import pandas as pd

from ml_returns_pred.rebalance_dates.rebalance_scheduler import RebalanceScheduler


class DatesRebalancer:
    def __init__(self, end_date: str, first_rebalancing_year: int, rebalancing_month: int,
                 rebalancing_week: int, rebalancing_weekday: Union[str, int], rebalancing_frequency: str):

        self.rebalance_scheduler = RebalanceScheduler(
            end_date=end_date,
            first_rebalancing_year=first_rebalancing_year,
            rebalancing_month=rebalancing_month,
            rebalancing_week=rebalancing_week,
            rebalancing_weekday=rebalancing_weekday,
            rebalancing_frequency=rebalancing_frequency
        )

        self.rebalancing_dates: list[datetime] = self.rebalance_scheduler.generate_rebalancing_dates()

    def filter_dates_post_start(self, data_start_date: datetime):
        """
        Filters rebalancing dates to keep only those that are on or after the data start date.

        Parameters:
        - data_start_date (datetime): The start date of the data.
        """
        self.rebalancing_dates = [date for date in self.rebalancing_dates if date >= data_start_date]

    def align_to_rebalancing_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Rebalances the DataFrame to keep only the entries at the rebalancing dates.
        Drops all other entries and uses the closest next valid date if a specific rebalancing date is not present.
        """
        self.filter_dates_post_start(data.index.min())

        aligned_data_list = []
        data_index = data.index

        for date in self.rebalancing_dates:
            if date in data_index:
                aligned_data_list.append(data.loc[[date]])
            else:
                next_valid_date = data_index[data_index > date].min()
                if pd.isna(next_valid_date):
                    continue
                aligned_data_list.append(data.loc[[next_valid_date]])

        aligned_data = pd.concat(aligned_data_list)

        return aligned_data[~aligned_data.index.duplicated()]

    @staticmethod
    def find_closest_common_date(index_1: pd.DatetimeIndex, index_2: pd.DatetimeIndex,
                                 target_date: datetime) -> Union[pd.Timestamp, pd.NaT]:
        """
        Helper function to find the smallest common date that is closest to the target_date from two lists of dates.
        """
        common_dates = index_1.intersection(index_2)
        if not common_dates.empty:
            future_common_dates = common_dates[common_dates >= target_date]
            if not future_common_dates.empty:
                closest_common_date = future_common_dates.min()
                if closest_common_date != target_date:
                    print(f"Rebalancing date: {target_date} - Closest common date: {closest_common_date}")
                return closest_common_date
        return pd.NaT  # Return Not a Time if there is no common date

    def align_signals_to_common_dates(self, data_1: pd.DataFrame, data_2: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns two DataFrames to the closest common rebalancing dates based on self.rebalancing_dates.

        Parameters
        ----------
        data_1 : DataFrame
            A DataFrame containing the signal values for the first set of positions.
        data_2 : DataFrame
            A DataFrame containing the signal values for the second set of positions.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            A tuple containing two DataFrames: data_1 and data_2 aligned at the closest common rebalancing dates.
        """
        aligned_data_1_list = []
        aligned_data_2_list = []

        data_1_index = data_1.index
        data_2_index = data_2.index

        for rebalance_date in self.rebalancing_dates:
            closest_common_date = self.find_closest_common_date(
                index_1=data_1_index,
                index_2=data_2_index,
                target_date=rebalance_date
            )

            if pd.notna(closest_common_date):
                aligned_data_1_list.append(data_1.loc[[closest_common_date]])
                aligned_data_2_list.append(data_2.loc[[closest_common_date]])

        aligned_data_1 = pd.concat(aligned_data_1_list)
        aligned_data_2 = pd.concat(aligned_data_2_list)

        return (aligned_data_1[~aligned_data_1.index.duplicated()],
                aligned_data_2[~aligned_data_2.index.duplicated()])

    @staticmethod
    def filter_data_by_dates(index_dates: list[datetime], data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to keep only the signals at the specified index dates.

        Parameters:
        - index_dates (List[datetime]): A list of dates to keep in the DataFrame.
        - data (pd.DataFrame): The DataFrame to filter.

        Returns
        - pd.DataFrame: The filtered DataFrame containing only the specified index dates.
        """
        filtered_data = data.loc[index_dates]
        return filtered_data

    def format_dates_to_string(self, date_format: str = "%Y-%m-%d") -> list[str]:
        """
        Formats the rebalancing dates from datetime to string according to the specified format.

        Parameters
        - date_format (str): The string format to convert dates into (default is ISO format: YYYY-MM-DD).

        Returns
        - List[str]: A list of dates converted to the specified string format.
        """
        formatted_dates = [date.strftime(date_format) for date in self.rebalancing_dates]
        return formatted_dates

    @staticmethod
    def reindex_and_forward_fill_data(*dataframes: pd.DataFrame, reindex_frequency: str = 'B',
                                      fill_method: Literal["backfill", "bfill", "ffill", "pad"] = 'ffill',
                                      ) -> list[pd.DataFrame]:
        """
        Reindexes the given dataframes to the specified frequency and forward fills the values.

        Parameters
        ----------
        reindex_frequency : str, optional
            The frequency of the date range to reindex the dataframes (default is 'B' for business days).
        fill_method : str, optional
            Method to forward fill ('ffill' or 'bfill', default is 'ffill').
        *dataframes : pd.DataFrame
            Variable number of DataFrames to reindex and forward fill.

        Returns
        -------
        List[pd.DataFrame]
            A list of reindexed and forward filled DataFrames.
        """
        reindexed_filled_dataframes = []
        for dataframe in dataframes:
            all_dates = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq=reindex_frequency)
            reindexed_dataframe = dataframe.reindex(all_dates).fillna(method=fill_method)
            reindexed_filled_dataframes.append(reindexed_dataframe)
        return reindexed_filled_dataframes


if __name__ == "__main__":
    pass