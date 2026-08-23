
import pandas as pd


class DataPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the index of the DataFrame is a DateTimeIndex.
        Converts the index to DateTimeIndex if it's not.

        Parameters:
        data (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with DateTimeIndex.
        """
        print("Old index type:", type(data.index))
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, errors='coerce', format='%Y-%m-%d')
            print("New index type:", type(data.index))
        return data

    @staticmethod
    def forward_fill_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fills the data for each column between the first and last non-NaN values.

        Parameters:
        data (pd.DataFrame): The input DataFrame with potential NaN values.

        Returns:
        pd.DataFrame: DataFrame with forward-filled values.
        """
        missing_values = {}
        data = data.copy()  # Work on a copy to avoid modifying the original DataFrame

        for column in data.columns:
            col_data = data[column]
            non_nan_indices = col_data.dropna().index
            if not non_nan_indices.empty:
                start, end = non_nan_indices[0], non_nan_indices[-1]
                # Count missing values between the first and last non-NaN values
                missing_count = col_data.loc[start:end].isnull().sum()
                missing_values[column] = missing_count
                # Perform forward fill
                data.loc[start:end, column] = col_data.loc[start:end].ffill()

        # Display the number of missing values per column in the specified range
        # print("Number of missing values per column between the first and last non-NaN values:")
        # for column, count in missing_values.items():
        #     print(f"{column}: {count}")

        return data

    @staticmethod
    def align_dataframes_within_common_period(dataframe_1: pd.DataFrame, dataframe_2: pd.DataFrame,
                                              drop_na: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns the indices of two DataFrames to the common period between them
        and returns the aligned DataFrames. The DataFrames will be cut to start and end
        on the same dates, without reindexing the entire period.

        Parameters:
        dataframe_1 (pd.DataFrame): The first DataFrame.
        dataframe_2 (pd.DataFrame): The second DataFrame.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the two aligned DataFrames.
        """
        if drop_na:
            dataframe_1 = dataframe_1.dropna(how='all')
            dataframe_2 = dataframe_2.dropna(how='all')

        # Determine the latest start date and the earliest end date between the two DataFrames
        start_date = max(dataframe_1.index.min(), dataframe_2.index.min())
        end_date = min(dataframe_1.index.max(), dataframe_2.index.max())

        print("Common period start date:", start_date)
        print("Common period end date:", end_date)

        # Use loc to restrict each DataFrame to the common period
        aligned_dataframe_1 = dataframe_1.loc[start_date:end_date]
        aligned_dataframe_2 = dataframe_2.loc[start_date:end_date]

        return aligned_dataframe_1, aligned_dataframe_2

    def keep_data_until_max_date(self, data: pd.DataFrame, max_date: str) -> pd.DataFrame:
        """
        Keeps the data up to the specified maximum date and returns the truncated DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        max_date (str): The maximum date up to which to keep the data.

        Returns:
        pd.DataFrame: DataFrame with data up to the maximum date.
        """
        data = self.ensure_datetime_index(data=data)
        data = data.loc[data.index <= max_date]
        return data

    @staticmethod
    def convert_datetime_index_to_month_end(data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the index from the first day of the month to the last day of the previous month
        and sets the frequency to month-end (ME).

        Parameters:
        data (pd.DataFrame): The input DataFrame with monthly data indexed by the first day of the month.

        Returns:
        pd.DataFrame: DataFrame with index shifted to the last day of the month.
        """

        data.index = data.index.to_period('M').to_timestamp('M') - pd.offsets.MonthEnd(1)
        data.index.freq = 'ME'

        return data

    @staticmethod
    def drop_columns_with_missing_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns with missing data from the DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with columns containing missing data dropped.
        """
        return data.dropna(axis=1)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the preprocessing steps in logical order and returns the preprocessed DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        data = self.ensure_datetime_index(data=data)
        data = self.forward_fill_data(data=data)
        return data

    def preprocess_macro_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts index from the first day of the month to the last day of the previous month
        and sets the frequency to month-end (ME).

        Parameters:
        data (pd.DataFrame): The input DataFrame with monthly data indexed by the first day of the month.

        Returns:
        pd.DataFrame: DataFrame with index shifted to the last day of the month.
        """
        data = self.ensure_datetime_index(data=data)
        # data = self.convert_datetime_index_to_month_end(data=data)
        data = self.drop_columns_with_missing_data(data=data)
        return data


if __name__ == '__main__':
    from ml_returns_pred.read_data.data_reader import DataReader

    dr = DataReader()
    relative_data_path = "../../data/raw_data/canadian_stocks_data_10.csv"
    relative_macro_data_path = "../../data/raw_data/macro_data_vif.csv"
    prices_data = dr.read_single_columns_level_data(relative_file_path=relative_data_path, index_col=0)
    macro_data = dr.read_single_columns_level_data(relative_file_path=relative_macro_data_path, index_col=0)
    print(prices_data.head(8))

    dp = DataPreprocessor()

    prices_data_preprocessed = dp.preprocess(data=prices_data)
    macro_data_preprocessed = dp.preprocess_macro_data(data=macro_data)
    print(prices_data_preprocessed.head(8))
    print(macro_data_preprocessed.head(8))
    print(macro_data_preprocessed.index.freq)


