import os
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from ml_returns_pred.file_management.file_manager import FileManagerDynamic


class DataDownloader:
    """A class to download and manage stock and benchmark data."""

    def __init__(self, start_date: dict, end_date: dict, column_to_keep: str = 'Close',
                 interval: str = '1d', group_by: str = 'ticker'):
        """
        Initialize the DataDownloader with common parameters for both stock and benchmark data.

        Parameters:
        start_date (dict): Dictionary containing 'year', 'month', and 'day' for the start date.
        end_date (dict): Dictionary containing 'year', 'month', and 'day' for the end date.
        column_to_keep (str): Column to keep from the downloaded data. Default is 'Close'.
        interval (str): Interval for the stock data. Default is '1d'.
        """
        self.start_date = self.parse_date(**start_date)
        self.end_date = self.parse_date(**end_date)
        self.column_to_keep = column_to_keep
        self.interval = interval
        self.group_by = group_by
        self.file_manager = FileManagerDynamic()

    @staticmethod
    def parse_date(**kwargs: dict[str, int]) -> datetime:
        """
        Convert keyword arguments to a datetime object.

        Parameters:
        **kwargs (Dict[str, int]): Dictionary containing 'year', 'month', and 'day'.

        Returns:
        datetime: Parsed datetime object.
        """
        return datetime(
            year=kwargs.get('year', datetime.now().year),
            month=kwargs.get('month', 1),
            day=kwargs.get('day', 1)
        )

    @staticmethod
    def _download_data_from_yfinance(tickers: list[str] | str, start_str: str, end_str: str,
                                     interval: str, **kwargs: Any) -> pd.DataFrame:
        """
        Download stock data using yfinance with additional kwargs for flexibility.

        Parameters:
        tickers (List[str]): List of stock tickers to download.
        start_str (str): Start date as a string.
        end_str (str): End date as a string.
        interval (str): Interval for the stock data.
        **kwargs (Any): Additional keyword arguments for yfinance download.

        Returns:
        pd.DataFrame: Raw downloaded data.
        """
        return yf.download(tickers=tickers, start=start_str, end=end_str, interval=interval, **kwargs)

    def extract_column(self, data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """
        Extract specified column from the downloaded data.

        Parameters:
        data (pd.DataFrame): Raw downloaded data.
        tickers (List[str]): List of stock tickers.

        Returns:
        pd.DataFrame: DataFrame containing specified column for each ticker.
        """
        return pd.DataFrame({ticker: data[ticker][self.column_to_keep] for ticker in tickers})

    def save_data(self, data: pd.DataFrame, start_str: str, end_str: str, file_name: str):
        """
        Save the data to a file using FileManagerDynamic.

        Parameters:
        data (pd.DataFrame): Data to save.
        start_str (str): Start date as a string.
        end_str (str): End date as a string.
        file_name (str): Name of the file to save.
        """
        self.file_manager.save_data(
            folder_name='raw_data',
            file_name=f'{file_name}_{start_str}_to_{end_str}.csv',
            data=data
        )

    def file_exists(self, file_name: str, start_str: str, end_str: str) -> bool:
        """
        Check if a file exists using FileManagerDynamic.

        Parameters:
        file_name (str): Name of the file.
        start_str (str): Start date as a string.
        end_str (str): End date as a string.

        Returns:
        bool: True if the file exists, False otherwise.
        """
        file_path = f'{file_name}_{start_str}_to_{end_str}.csv'
        return self.file_manager.search(target_name=file_path, start_path=os.getcwd(), search_type='file') is not None

    def download_stock_data(self, tickers: list[str], stocks_file_name: str = 'canadian_stocks', **kwargs: Any):
        """
        Download stock data for the given tickers from start_date to end_date and save the DataFrame.

        Parameters:
        tickers (List[str]): List of stock tickers to download.
        stocks_file_name (str): File name to save the stock data. Default is 'canadian_stocks'.
        **kwargs (Any): Additional keyword arguments for yfinance download.
        """
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')

        if self.file_exists(stocks_file_name, start_str, end_str):
            print(f"Data already downloaded: {stocks_file_name}_{start_str}_to_{end_str}.csv")
            return

        kwargs.pop('interval', None)  # Remove 'interval' if it exists in kwargs

        data = self._download_data_from_yfinance(
            tickers=tickers, start_str=start_str, end_str=end_str,
            interval=self.interval, group_by=self.group_by, **kwargs
        )
        extracted_data = self.extract_column(data=data, tickers=tickers)

        self.save_data(
            data=extracted_data, start_str=start_str, end_str=end_str, file_name=stocks_file_name)

    def download_benchmark_data(self, benchmark_file_name: str = 'sptsx', benchmark_ticker: str = "^GSPTSE", **kwargs: Any):
        """
        Download benchmark data for the SPTSX (^GSPTSE) from start_date to end_date and save the DataFrame.

        Parameters:
        benchmark_file_name (str): File name to save the benchmark data. Default is 'sptsx'.
        benchmark_ticker (str): Ticker for the benchmark. Default is "^GSPTSE".
        **kwargs (Any): Additional keyword arguments for yfinance download.
        """
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')

        if self.file_exists(benchmark_file_name, start_str, end_str):
            print(f"Data already downloaded: {benchmark_file_name}_{start_str}_to_{end_str}.csv")
            return

        kwargs.pop('interval', None)  # Remove 'interval' if it exists in kwargs

        # Download the benchmark data
        data = self._download_data_from_yfinance(
            tickers=benchmark_ticker, start_str=start_str, end_str=end_str, interval=self.interval, **kwargs
        )

        # Extract the 'Close' column
        benchmark_prices = data[self.column_to_keep]

        # Normalize the index
        benchmark_prices.index = benchmark_prices.index.normalize()

        # change the column name to the benchmark_file_name
        benchmark_prices.name = benchmark_file_name

        self.save_data(data=benchmark_prices, start_str=start_str, end_str=end_str, file_name=benchmark_file_name)


if __name__ == '__main__':
    # Example usage
    tickers = ["ABX.TO", "AEM.TO", "ATD.TO", "BB.TO", "BBD-B.TO", "BCE.TO", "BMO.TO", "BN.TO", "BLDP.TO", "BNS.TO",
               "CAE.TO", "CCA.TO", "CCL-B.TO", "CCO.TO", "CM.TO", "CNR.TO", "CTC.TO", "CTC-A.TO", "EMA.TO", "EMP-A.TO",
               "ENGH.TO", "ENB.TO", "FTS.TO", "FTT.TO", "GIL.TO", "HR-UN.TO", "IDG.TO", "IMO.TO", "L.TO", "MFC.TO", "MFI.TO",
               "MRU.TO", "MTL.TO", "NA.TO", "ONEX.TO", "POW.TO", "RCI-B.TO", "RY.TO", "SAP.TO", "SJ.TO",
               "STN.TO", "SU.TO", "T.TO", "TCL-A.TO", "TECK-B.TO", "TRP.TO", "TD.TO", "WN.TO", "WDO.TO", "XIU.TO"]

    start_date_dict = {'year': 1999, 'month': 11, 'day': 1}
    end_date_dict = {'year': 2024, 'month': 6, 'day': 1}

    data_downloader = DataDownloader(start_date=start_date_dict, end_date=end_date_dict)
    data_downloader.download_stock_data(tickers=tickers, group_by='ticker', auto_adjust=True)
    data_downloader.download_benchmark_data(auto_adjust=True)
