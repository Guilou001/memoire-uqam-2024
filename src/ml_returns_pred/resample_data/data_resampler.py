from typing import Union

import pandas as pd

from ml_returns_pred.rebalance_dates.dates_rebalancer import DatesRebalancer


class DataResampler:
    @staticmethod
    def ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the index of the data is a datetime index.

        Parameters:
            data (pd.DataFrame): The data to process.

        Returns:
            pd.DataFrame: Data with a datetime index.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index, errors='coerce')
            except Exception as e:
                raise ValueError(f"Failed to convert index to datetime: {e}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data by ensuring a datetime index and converting to floats.

        Parameters:
            data (pd.DataFrame): The data to prepare.

        Returns:
            pd.DataFrame: Prepared data.
        """
        data = DataResampler.ensure_datetime_index(data)
        return data.apply(pd.to_numeric, errors='coerce')

    @staticmethod
    def _extract_date_info(data: pd.DataFrame) -> tuple:
        """
        Extracts the first rebalancing year, rebalancing month, and end date year from the data.

        Returns:
            tuple: (first_rebalancing_year, rebalancing_month, end_date_year)
        """
        first_date = data.index.min()
        last_date = data.index.max()

        first_rebalancing_year = first_date.year
        rebalancing_month = first_date.month
        end_date_year = last_date.year

        return first_rebalancing_year, rebalancing_month, end_date_year

    def resample_data(self, data: pd.DataFrame, resample_week: int, resample_weekday: Union[str, int],
                      resample_frequency: str) -> pd.DataFrame:
        """
        Resamples the data based on rebalancing dates specified.

        Parameters:
            data (pd.DataFrame): The data to resample.
            resample_week (int): The week to rebalance.
            resample_weekday (Union[str, int]): The weekday to rebalance.
            resample_frequency (str): The frequency of rebalancing.

        Returns:
            pd.DataFrame: Resampled data.
        """
        data = self._prepare_data(data)
        first_resampling_year, resampling_month, end_date_year = self._extract_date_info(data)

        rebalancer = DatesRebalancer(
            end_date=f"{end_date_year}-12-31",
            first_rebalancing_year=first_resampling_year,
            rebalancing_month=resampling_month,
            rebalancing_week=resample_week,
            rebalancing_weekday=resample_weekday,
            rebalancing_frequency=resample_frequency
        )

        resampled_data = rebalancer.align_to_rebalancing_dates(data=data)
        return resampled_data.dropna(axis=0, how='all')

    def resample_and_forward_fill(self, data_to_resample: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the index of data_to_resample to match the index of reference_data,
        forward fills the missing values, and applies the frequency of reference_data to data_to_resample.

        Parameters:
            data_to_resample (pd.DataFrame): The DataFrame to be resampled.
            reference_data (pd.DataFrame): The reference DataFrame whose index will be used for resampling.

        Returns:
            pd.DataFrame: Resampled and forward-filled data.
        """
        # Ensure both dataframes have datetime index
        data_to_resample = self.ensure_datetime_index(data_to_resample)
        reference_data = self.ensure_datetime_index(reference_data)

        # Intersection of indices
        common_index = data_to_resample.index.union(reference_data.index)

        # Forward fill the missing values in data_to_resample
        data_to_resample = data_to_resample.reindex(common_index).ffill()

        # Reindex data_to_resample based on the index of reference_data and apply the frequency
        resampled_data = data_to_resample.reindex(reference_data.index)
        resampled_data.index.freq = reference_data.index.freq

        return resampled_data.dropna(axis=0, how='all')

    def convert_datetime_index_to_period(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Converts the datetime index of the data to a PeriodIndex with the specified frequency.

        Parameters:
            data (pd.DataFrame): The data to process.
            freq (str): The frequency of the PeriodIndex.

        Returns:
            pd.DataFrame: Data with a PeriodIndex.
        """
        data = self.ensure_datetime_index(data)
        data.index = pd.PeriodIndex(data.index, freq=freq)

        return data

    def specify_datetime_index_frequency(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Specifies the frequency of the datetime index of the data.

        Parameters:
            data (pd.DataFrame): The data to process.
            freq (str): The frequency of the datetime index.

        Returns:
            pd.DataFrame: Data with a specified frequency of the datetime index.
        """
        data = self.ensure_datetime_index(data)

        # Reindex the data to the specified frequency
        data = data.asfreq(freq=freq, method='ffill')

        return data


if __name__ == "__main__":
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.read_data.data_reader import DataReader

    dr = DataReader()
    relative_data_path = "../../data/raw_data/canadian_stocks_data.csv"
    relative_macro_data_path = "../../data/raw_data/macro_data.csv"
    prices_data = dr.read_single_columns_level_data(relative_file_path=relative_data_path, index_col=0)
    macro_data = dr.read_single_columns_level_data(relative_file_path=relative_macro_data_path, index_col=0)
    print(prices_data.head(8))

    dp = DataPreprocessor()
    prices_data_preprocessed = dp.preprocess(data=prices_data)
    macro_data_preprocessed = dp.preprocess_macro_data(data=macro_data)

    # prices_data_aligned, macro_data_aligned = dp.align_dataframes_within_common_period(
    #     dataframe_1=prices_data_preprocessed,
    #     dataframe_2=macro_data_preprocessed
    # )

    dr = DataResampler()

    prices_data_resampled = dr.resample_and_forward_fill(
        data_to_resample=prices_data_preprocessed,
        reference_data=macro_data_preprocessed
    )

    print(prices_data_resampled.head(8))
    print(prices_data_resampled.index.freq)





