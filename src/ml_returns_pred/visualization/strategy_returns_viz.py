import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


class StrategyReturnsViz:
    """
    A class to visualize strategy returns stored in a DataFrame.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the strategy returns with dates as the index.
        benchmarks (List[pd.Series]): List of benchmark series to compare against the strategies.

    Methods:
        plot_returns(title: str = 'Strategy Returns', xlabel: str = 'Date', ylabel: str = 'Return',
                     save_path: str = None, subplots_kwargs: dict = None):
            Plots the strategy returns.
        compute_cumulative_returns(df: pd.DataFrame, method: str = 'arithmetic') -> pd.DataFrame:
            Computes the cumulative returns using the specified method.
        plot_cumulative_returns(method: str = 'arithmetic', title: str = None, xlabel: str = 'Date',
                                ylabel: str = 'Cumulative Return', save_path: str = None, subplots_kwargs: dict = None):
            Plots the cumulative returns using the specified method.
        align_period():
            Aligns the period of the DataFrame and benchmarks to a common period.
    """

    def __init__(self, df: pd.DataFrame, benchmark_returns_paths: list[str] | None = None):
        """
        Initializes the StrategyReturnsViz with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the strategy returns with dates as the index.
            benchmark_returns_paths (List[str], optional): List of paths to CSV files containing the benchmark data.
        """
        self.df = self._validate_dataframe(df)
        self.benchmarks = self._load_benchmarks(benchmark_returns_paths) if benchmark_returns_paths else []

        self._align_period()

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and preprocesses the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            pd.DataFrame: The validated DataFrame.
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.columns = StrategyReturnsViz._preprocess_column_names(df.columns)
        return df

    @staticmethod
    def _load_benchmarks(paths: list[str]) -> list[pd.Series]:
        """
        Loads benchmark data from the provided file paths.

        Args:
            paths (List[str]): List of file paths to CSV files containing benchmark data.

        Returns:
            List[pd.Series]: List of benchmark series.
        """
        benchmarks = []
        for path in paths:
            benchmark = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            benchmarks.append(benchmark)
        return benchmarks

    def _align_period(self) -> None:
        """
        Aligns the period of the DataFrame and benchmarks.
        Filters benchmarks to start from the first date in self.df.index and
        to end at the last date in self.df.index.
        """
        if self.benchmarks:
            first_date = self.df.index.min()
            last_date = self.df.index.max()

            # Filter benchmarks to the period of self.df
            self.benchmarks = [benchmark.loc[first_date:last_date] for benchmark in self.benchmarks]

    @staticmethod
    def _preprocess_column_names(columns: pd.Index) -> list[str]:
        """
        Preprocesses column names by removing everything after _regressor or _classifier,
        but keeping _regressor and _classifier themselves.

        Args:
            columns (pd.Index): The original column names.

        Returns:
            List[str]: The preprocessed column names.
        """
        pattern = re.compile(r'(_regressor|_classifier).*')
        return [pattern.sub(r'\1', col) for col in columns]

    def plot_returns(self, title: str = 'Strategy Returns', xlabel: str = 'Date',
                     ylabel: str = 'Return', save_path: str = None, subplots_kwargs: dict = None) -> None:
        """
        Plots the strategy returns.

        Args:
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            save_path (str, optional): The path to save the plot image. If None, the plot is not saved.
            subplots_kwargs (dict, optional): Additional arguments for plt.subplots.
        """
        if subplots_kwargs is None:
            subplots_kwargs = {}

        self._plot_individual_strategy('_regressor', title, xlabel, ylabel, save_path, subplots_kwargs)
        self._plot_individual_strategy('_classifier', title, xlabel, ylabel, save_path, subplots_kwargs)

    def _plot_individual_strategy(self, strategy_type: str, title: str, xlabel: str, ylabel: str,
                                  save_path: str | None, subplots_kwargs: dict) -> None:
        """
        Plots individual strategy types (regressor or classifier).

        Args:
            strategy_type (str): The strategy type to plot ('_regressor' or '_classifier').
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            save_path (str, optional): The path to save the plot image. If None, the plot is not saved.
            subplots_kwargs (dict, optional): Additional arguments for plt.subplots.
        """
        fig, ax = plt.subplots(**subplots_kwargs)

        for column in self.df.columns:
            if strategy_type in column:
                ax.plot(self.df.index, self.df[column], label=column)

        for i, benchmark in enumerate(self.benchmarks):
            ax.plot(benchmark.index, benchmark, label=f'{benchmark.name}', linestyle='--')

        ax.set_title(f"{title} ({strategy_type.capitalize().replace('_', '')})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

        if save_path:
            fig.savefig(f"{save_path}{strategy_type}.png")

        plt.show()

    @staticmethod
    def compute_cumulative_returns(df: pd.DataFrame, method: str = 'arithmetic') -> pd.DataFrame:
        """
        Computes the cumulative returns using the specified method.

        Args:
            df (pd.DataFrame): The DataFrame containing the returns.
            method (str): The method for computing cumulative returns ('arithmetic' or 'logarithmic').

        Returns:
            pd.DataFrame: The DataFrame containing the cumulative returns.
        """
        if method == 'arithmetic':
            cumulative_returns = (1 + df).cumprod() - 1
        elif method == 'logarithmic':
            # Compute the cumulative sum of logarithmic returns and then exponentiate
            cumulative_returns = np.log1p((1 + df).cumprod() - 1)
        else:
            raise ValueError("method must be 'arithmetic' or 'logarithmic'")
        return cumulative_returns

    def plot_cumulative_returns(self, method: str = 'arithmetic', title: str = None, xlabel: str = 'Date',
                                ylabel: str = 'Cumulative Return', save_path: str = None,
                                subplots_kwargs: dict = None, log_scale: bool = False) -> None:
        """
        Plots the cumulative returns using the specified method.

        Args:
            method (str): The method for computing cumulative returns ('arithmetic' or 'logarithmic').
            title (str): The title of the plot. If None, a default title is used based on the method.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            save_path (str, optional): The path to save the plot image. If None, the plot is not saved.
            subplots_kwargs (dict, optional): Additional arguments for plt.subplots.
            log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis.
        """
        if subplots_kwargs is None:
            subplots_kwargs = {}

        cumulative_returns = self.compute_cumulative_returns(self.df, method)
        # Shift cumulative returns up by 1 to ensure positivity for log scale
        if log_scale:
            cumulative_returns = cumulative_returns + 1

        self._plot_individual_cumulative_returns(cumulative_returns, method, '_regressor', title, xlabel, ylabel,
                                                 save_path, subplots_kwargs, log_scale)
        self._plot_individual_cumulative_returns(cumulative_returns, method, '_classifier', title, xlabel, ylabel,
                                                 save_path, subplots_kwargs, log_scale)

    def _plot_individual_cumulative_returns(
            self,
            cumulative_returns: pd.DataFrame,
            method: str,
            strategy_type: str,
            title: str | None,
            xlabel: str,
            ylabel: str,
            save_path: str | None,
            subplots_kwargs: dict,
            log_scale: bool
    ) -> None:
        """
        Plots individual cumulative returns for strategy types (regressor or classifier).

        Args:
            cumulative_returns (pd.DataFrame): The DataFrame containing the cumulative returns.
            method (str): The method for computing cumulative returns ('arithmetic' or 'logarithmic').
            strategy_type (str): The strategy type to plot ('_regressor' or '_classifier').
            title (str, optional): The title of the plot. If None, a default title is used based on the method.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            save_path (str, optional): The path to save the plot image. If None, the plot is not saved.
            subplots_kwargs (dict, optional): Additional arguments for plt.subplots.
            log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis.
        """
        fig, ax = plt.subplots(**subplots_kwargs)

        if log_scale:
            scale_type = 'logarithmic'
        else:
            scale_type = 'arithmetic'

        for column in cumulative_returns.columns:
            if strategy_type in column:
                ax.plot(cumulative_returns.index, cumulative_returns[column], label=column)

        for i, benchmark in enumerate(self.benchmarks):
            benchmark_cumulative = self.compute_cumulative_returns(benchmark.to_frame(), method)

            # Shift benchmark cumulative returns up by 1 if using log scale
            if log_scale:
                benchmark_cumulative = benchmark_cumulative + 1

            ax.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.squeeze(),
                label=f'{benchmark.name}',
                linestyle='--'
            )

        if title is None:
            title = f'Cumulative Returns ({strategy_type.capitalize().replace("_", "")})'
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()
        ax.grid(True)

        formatter = mticker.ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())  # Formatteur pour garder les valeurs numériques
            ax.yaxis.get_major_formatter().set_scientific(False)  # Désactive la notation scientifique
            ax.yaxis.get_major_formatter().set_useOffset(False)  # Désactive l'offset

        if save_path:
            fig.savefig(f"{save_path}{strategy_type}.png")

        plt.show()


if __name__ == '__main__':
    from ml_returns_pred.file_management.file_manager import FileManagerStatic

    # increase the number of columns to display
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    folder_path = "../../data/concatenated_data/compute_strategy_returns/strategy_returns_concat.csv"

    fm = FileManagerStatic()

    strategy_returns = fm.load_data(relative_file_path=folder_path, index_col=0)

    benchmark_paths = [
        "../../data/intermediate_data/benchmark_returns/NASDAQ_returns.csv",
        "../../data/intermediate_data/benchmark_returns/equally_weighted_strategy_returns.csv"
    ]

    sv = StrategyReturnsViz(df=strategy_returns, benchmark_returns_paths=benchmark_paths)

    sv.plot_cumulative_returns(method='arithmetic')
