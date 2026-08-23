import pandas as pd

from ml_returns_pred.file_management.file_manager import FileManagerStatic


class DataReader:
    def __init__(self):
        self.file_manager_static = FileManagerStatic()

    def read_double_columns_level_data(self, relative_file_path: str, drop_second_level_column: bool = True,
                                       **kwargs) -> (
            pd.DataFrame):
        """
        Reads a double index level data file.

        Parameters:
            relative_file_path (str): The name of the file to read.
            drop_second_level_column (bool): Whether to drop the second level column.

        Returns:
            pd.DataFrame: The data.
        """
        data = self.file_manager_static.load_data(relative_file_path=relative_file_path, **kwargs)

        if drop_second_level_column:
            data.columns = data.columns.droplevel(1)
        return data

    def read_single_columns_level_data(self, relative_file_path: str, **kwargs) -> pd.DataFrame:
        """
        Reads a single index level data file.

        Parameters:
            relative_file_path (str): The name of the file to read.

        Returns:
            pd.DataFrame: The data.
        """
        data = self.file_manager_static.load_data(relative_file_path=relative_file_path, **kwargs)
        return data

    def read_risk_free_rates_data(self, relative_file_path: str, rebalancing_frequency: str = "M",
                                  **kwargs) -> pd.DataFrame:
        """
        Reads a risk free rates data file.

        Parameters:
            relative_file_path (str): The name of the file to read.
            rebalancing_frequency (str): The rebalancing frequency.

        Returns:
            pd.DataFrame: The data.
        """
        frequency_mapping = {"M": "1 month", "Q": "3 months", "S": "6 months", "A": "1 year"}

        data = self.file_manager_static.load_data(relative_file_path=relative_file_path, **kwargs)[
            [frequency_mapping[rebalancing_frequency]]]
        return data


if __name__ == '__main__':
    dr = DataReader()
    relative_data_path = "../../data/raw_data/W-FRI_20000107_20230602_SPTSX Index_TOT_RETURN_INDEX_GROSS_DVDS.csv"
    prices_data = dr.read_double_columns_level_data(relative_file_path=relative_data_path, header=[0, 1], index_col=0)
    print(prices_data.head(8))
