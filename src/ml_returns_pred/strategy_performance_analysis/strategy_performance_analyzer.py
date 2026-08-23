import os
import warnings
import webbrowser
from typing import Union

import numpy as np
import pandas as pd
import quantstats_lumi as qs

from ml_returns_pred.file_management.file_manager import FileManagerDynamic

warnings.simplefilter(action='ignore', category=FutureWarning)


class StrategyPerformanceAnalyzer:
    """
    A class for analyzing the performance of a portfolio against a benchmark.

    This class provides methods for preparing the portfolio and benchmark returns for analysis,
    generating backtesting reports, and computing performance metrics. The reports can be generated in two formats:
    html and full report. The performance metrics can be computed in three modes: 'full', 'simple', and 'stats'.

    Attributes
    ----------
    portfolio_returns : pd.DataFrame
        A DataFrame containing the returns of the portfolio. It is prepared for analysis in the constructor.
    benchmark_returns : pd.DataFrame
        A DataFrame containing the returns of the benchmark. It is prepared for analysis in the constructor.

    Methods
    -------
    _prepare_portfolio_returns(calc_portfolio_returns: pd.DataFrame) -> pd.DataFrame:
        Prepares the portfolio returns DataFrame for analysis.
    _prepare_benchmark_returns(benchmark_returns: pd.DataFrame) -> pd.DataFrame:
        Prepares the benchmark returns DataFrame for analysis.
    _move_file_to_directory(file_path: str, dest_directory: str) -> None:
        Moves a file to a specified destination directory.
    generate_backtesting_report_html(rf: float, title: str, grayscale: bool = False, output: bool = True, match_dates:
    bool = True, open_in_browser: bool = False) -> None:
        Generates an HTML backtesting report.
    generate_backtesting_report_full(rf: float, grayscale: bool = False, match_dates: bool = True) -> None:
        Generates a full backtesting report.
    compute_performance_metrics(rf: float, mode: str = "full", prepare_returns: bool = False, match_dates: bool = True)
    -> None:
        Computes performance metrics for the portfolio against the benchmark.
    """

    def __init__(
        self,
        portfolio_returns: pd.DataFrame,
        benchmark_prices: Union[pd.DataFrame, None],
        strategy_name: str,
    ):
        self.portfolio_returns: pd.DataFrame = self._prepare_portfolio_returns(portfolio_returns=portfolio_returns)
        self.benchmark_returns: pd.Series = self._prepare_benchmark_returns(benchmark_prices=benchmark_prices)
        self.strategy_name: str = strategy_name

    @staticmethod
    def _prepare_portfolio_returns(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the portfolio returns DataFrame for analysis.

        Returns
        -------
        pd.DataFrame
            The portfolio returns DataFrame.
        """
        if portfolio_returns.index.dtype != "datetime64[ns]":
            portfolio_returns.index = pd.to_datetime(
                portfolio_returns.index, format="%Y-%m-%d"
            )

        # Name column "Portfolio_Returns"
        portfolio_returns.columns = ["Portfolio_Returns"]


        # Check for null values in "Portfolio_Returns" column
        nan_count = portfolio_returns["Portfolio_Returns"].isnull().sum()
        if nan_count >= 1:
            print(f"Number of NaN values detected: {nan_count}")

        if nan_count == 1:
            nan_index = portfolio_returns["Portfolio_Returns"].index[portfolio_returns["Portfolio_Returns"].isnull()][0]
            portfolio_returns = portfolio_returns.loc[portfolio_returns.index > nan_index]

        # Check for null values in "Portfolio_Returns" column
        if portfolio_returns["Portfolio_Returns"].isnull().any():
            raise ValueError("The 'Portfolio_Returns' column contains null values.")

        # Ensure we are working on a copy to avoid SettingWithCopyWarning
        portfolio_returns = portfolio_returns.copy()
        portfolio_returns["Portfolio_Returns"] = pd.to_numeric(portfolio_returns["Portfolio_Returns"])

        return portfolio_returns

    @staticmethod
    def _prepare_benchmark_returns(benchmark_prices: pd.DataFrame) -> Union[pd.Series, None]:
        """
        Prepare the benchmark returns DataFrame for analysis.

        Returns
        -------
        pd.Series
            The benchmark returns DataFrame.
        """
        if benchmark_prices is None:
            return benchmark_prices

        if benchmark_prices.index.dtype != "datetime64[ns]":
            benchmark_prices.index = pd.to_datetime(benchmark_prices.index)

        benchmark_returns = benchmark_prices.pct_change().dropna()

        return pd.to_numeric(pd.Series(benchmark_returns.iloc[:, 0], name=benchmark_prices.columns[0]))

    @staticmethod
    def _move_file_to_directory(src_folder: str, dest_folder: str, file_name: str) -> None:
        """
        Move a file to a specified destination directory.

        Parameters
        ----------
        src_folder : str
            The source folder containing the file.
        dest_folder : str
            The destination folder to move the file to.
        file_name : str
            The name of the file to move.

        Raises
        ------
        ValueError
            If either `file_path` or `dest_directory` is not a valid directory path.
        OSError
            If an error occurs while moving the file.
        """

        fmd = FileManagerDynamic()

        fmd.move_file(src_folder=src_folder,
                      dest_folder=dest_folder,
                      file_name=file_name)

    def generate_backtesting_report_html(
            self,
            rf: float,
            periods_per_year: int,
            grayscale: bool = False,
            output: str | None = None,
            match_dates: bool = True,
            open_in_browser: bool = False,
    ) -> None:
        output_file = f"{self.strategy_name}_backtesting_report.html"
        qs.reports.html(
            returns=self.portfolio_returns["Portfolio_Returns"],
            benchmark=self.benchmark_returns,
            rf=rf,
            periods_per_year=periods_per_year,
            title=f"{self.strategy_name} strategy",
            output=output_file,
            grayscale=grayscale,
            download_filename=f"{self.strategy_name}_{output}.html",
            match_dates=match_dates,
        )

        self._move_file_to_directory(
            src_folder=os.getcwd(),
            dest_folder="reports",
            file_name=output_file
        )

        if open_in_browser:
            file_path = f"../reports/{output_file}"
            self._open_html_in_browser(file_path=file_path)

        return None

    def generate_backtesting_alpha_report_html(
            self,
            rf: float,
            periods_per_year: int,
            grayscale: bool = False,
            match_dates: bool = True,
            open_in_browser: bool = False,
    ) -> None:

        # Match dates between portfolio and benchmark returns
        common_dates = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        self.portfolio_returns = self.portfolio_returns.loc[common_dates]
        self.benchmark_returns = self.benchmark_returns.loc[common_dates]

        # Calculate alpha returns and convert to DataFrame with specified column name
        alpha_returns = (self.portfolio_returns["Portfolio_Returns"] - self.benchmark_returns[
            self.benchmark_returns.columns[0]]).to_frame(name="Alpha_Returns")

        # Generate HTML report
        output_file = f"{self.strategy_name}_backtesting_alpha_report.html"
        qs.reports.html(
            returns=alpha_returns["Alpha_Returns"],
            benchmark=None,
            rf=rf,
            periods_per_year=periods_per_year,
            title=f"{self.strategy_name} alpha strategy",
            output=output_file,
            grayscale=grayscale,
            download_filename=f"{self.strategy_name}_backtesting_alpha_report.html",
            match_dates=match_dates,
        )

        # Move file to directory
        self._move_file_to_directory(
            src_folder=os.getcwd(),
            dest_folder="reports",
            file_name=output_file
        )

        if open_in_browser:
            # add 0.1 sec delay to ensure file is moved before opening

            file_path = f"../reports/{output_file}"
            self._open_html_in_browser(file_path=file_path)

        return None

    def generate_backtesting_report_full(
            self, rf: float, grayscale: bool = False, match_dates: bool = True
    ) -> None:
        qs.reports.full(
            returns=self.portfolio_returns["Portfolio_Returns"],
            benchmark=self.benchmark_returns,
            rf=rf,
            grayscale=grayscale,
            match_dates=match_dates,
        )

        return None

    def compute_performance_metrics(
            self,
            rf: float,
            mode: str = "full",
            prepare_returns: bool = False,
            match_dates: bool = True,
    ) -> None:
        qs.reports.metrics(
            returns=self.portfolio_returns["Portfolio_Returns"],
            benchmark=self.benchmark_returns,
            rf=rf,
            mode=mode,
            prepare_returns=prepare_returns,
            match_dates=match_dates,
        )

        return None

    @staticmethod
    def _open_html_in_browser(file_path: str) -> None:
        """
        Open the specified HTML file in browser.

        Parameters
        ----------
        file_path : str
            The path to the HTML file to open.
        """
        webbrowser.open(f"file://{os.path.abspath(file_path)}")

        return None

    @staticmethod
    def _compute_omega_ratio(returns: pd.Series, required_return: float = 0.0, periods: int = 252) -> float:
        """
        Compute the Omega ratio of a returns series.

        Parameters
        ----------
        returns : pd.Series
            The series of returns.
        required_return : float
            The required return threshold. Default is 0.0.
        periods : int
            Number of periods in a year based on frequency (e.g., 252 for daily returns).

        Returns
        -------
        float
            The Omega ratio.
        """
        if len(returns) < 2 or required_return <= -1:
            return np.nan

        if periods == 1:
            return_threshold = required_return
        else:
            return_threshold = (1 + required_return) ** (1.0 / periods) - 1

        returns_less_thresh = returns - return_threshold
        numer = returns_less_thresh[returns_less_thresh > 0.0].sum()
        denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum()

        if denom.sum() > 0.0:  # Use .sum() to resolve the ambiguity
            return numer / denom

        return np.nan

    @staticmethod
    def _compute_cumulative_returns(returns: pd.Series) -> pd.Series:
        """
        Compute the cumulative returns of a Series.

        Parameters
        ----------
        returns : pd.Series
            A Series containing the returns.

        Returns
        -------
        pd.Series
            A Series containing the cumulative returns.
        """
        return (1 + returns).prod(axis=0) - 1  # Specify axis=0 to avoid FutureWarning

    def get_key_performance_metrics(self, rf: float, periods_per_year: int, annualize: bool = True) -> pd.DataFrame:
        """
        Compute key performance metrics including CAGR, Sharpe ratio, and Volatility.

        Parameters
        ----------
        rf : float
            Risk-free rate per period.
        periods_per_year : int
            Number of periods in a year based on frequency (e.g., 252 for daily returns).
        annualize : bool
            Whether to annualize the Sharpe ratio and Volatility or not.

        Returns
        -------
        pd.DataFrame
            DataFrame containing key performance metrics.
        """
        portfolio_returns = self.portfolio_returns["Portfolio_Returns"]

        metrics = {
            "Cumulative_Returns": self._compute_cumulative_returns(returns=portfolio_returns),
            "CAGR": qs.stats.cagr(returns=portfolio_returns, rf=rf, periods=periods_per_year)*100,
            "Sharpe": qs.stats.sharpe(returns=portfolio_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "Volatility": qs.stats.volatility(returns=portfolio_returns, periods=periods_per_year, annualize=annualize),
            "Max_Drawdown": qs.stats.max_drawdown(prices=portfolio_returns),
            "Sortino": qs.stats.sortino(returns=portfolio_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "R_Squared": qs.stats.r_squared(returns=portfolio_returns, benchmark=self.benchmark_returns),
            "Omega_Ratio": self._compute_omega_ratio(returns=portfolio_returns, periods=periods_per_year),
        }

        return pd.DataFrame(metrics, index=[self.strategy_name]).T

    def get_key_performance_metrics_benchmark(self, benchmark_returns: pd.Series,
                                      rf: float, periods_per_year: int, annualize: bool = True) -> pd.DataFrame:
        """
        Compute key performance metrics including CAGR, Sharpe ratio, and Volatility.

        Parameters
        ----------
        benchmark_returns : pd.DataFrame
            DataFrame containing benchmark returns.
        rf : float
            Risk-free rate per period.
        periods_per_year : int
            Number of periods in a year based on frequency (e.g., 252 for daily returns).
        annualize : bool
            Whether to annualize the Sharpe ratio and Volatility or not.

        Returns
        -------
        pd.DataFrame
            DataFrame containing key performance metrics.
        """

        benchmark_returns = pd.Series(benchmark_returns)

        # take same period as self.portfolio_returns
        benchmark_returns = benchmark_returns.loc[self.portfolio_returns.index[0]:self.portfolio_returns.index[-1]]

        metrics = {
            "Cumulative_Returns": self._compute_cumulative_returns(returns=benchmark_returns),
            "CAGR": qs.stats.cagr(returns=benchmark_returns, rf=rf, periods=periods_per_year),
            "Sharpe": qs.stats.sharpe(returns=benchmark_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "Volatility": qs.stats.volatility(returns=benchmark_returns, periods=periods_per_year, annualize=annualize),
            "Max_Drawdown": qs.stats.max_drawdown(prices=benchmark_returns),
            "Sortino": qs.stats.sortino(returns=benchmark_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "R_Squared": 1,
            "Omega_Ratio": self._compute_omega_ratio(returns=benchmark_returns, periods=periods_per_year),
        }

        return pd.DataFrame(metrics, index=["Strategy"]).T



if __name__ == '__main__':
    from ml_returns_pred.compute_returns.from_prices_to_returns import FromPricesToReturns
    from ml_returns_pred.compute_strategy_returns.strategy_returns_calculator_optimized import (
        StrategyReturnsCalculatorOptimized,
    )
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
    benchmark_prices_path = "../../data/raw_data/sptsx.csv"

    prices_data = dr.read_single_columns_level_data(relative_file_path=relative_data_path, index_col=0)
    macro_data = dr.read_single_columns_level_data(relative_file_path=relative_macro_data_path, index_col=0)
    benchmark_prices = dr.read_single_columns_level_data(relative_file_path=benchmark_prices_path, index_col=0, sep=';')

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

    sp = StrategyPerformanceAnalyzer(
        portfolio_returns=portfolio_returns,
        benchmark_prices=benchmark_prices,
        strategy_name="lasso"
    )

    sp.generate_backtesting_report_html(
        rf=0.0,
        periods_per_year=252,
        grayscale=False,
        output="backtesting_report",
        match_dates=True,
        open_in_browser=True
    )
