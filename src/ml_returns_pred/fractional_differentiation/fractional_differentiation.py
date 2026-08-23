import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FractionalDifferentiation:
    def __init__(self):
        pass

    # @staticmethod
    # def fracdiff(data: pd.DataFrame, d: float) -> pd.DataFrame:
    #     """
    #     Fractionally differentiate the input data.
    #
    #     Parameters:
    #         data (pd.DataFrame): The input data to fractionally differentiate.
    #         d (float): The degree of differentiation.
    #
    #     Returns:
    #         pd.DataFrame: The fractionally differentiated data.
    #     """
    #     fracdiff = Fracdiff(d=d)
    #     transformed_data = fracdiff.fit_transform(data.values)
    #     return pd.DataFrame(transformed_data, index=data.index, columns=data.columns)

    # @staticmethod
    # def fracdiff_stat(data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Fractionally differentiate the input data using the statistical method.
    #
    #     Parameters:
    #         data (pd.DataFrame): The input data to fractionally differentiate.
    #
    #     Returns:
    #         pd.DataFrame: The fractionally differentiated data.
    #     """
    #     fracdiff_stat = FracdiffStat(n_jobs=-1)
    #     transformed_data = fracdiff_stat.fit_transform(data.values)
    #     return pd.DataFrame(transformed_data, index=data.index, columns=data.columns)

    @staticmethod
    def compute_returns(data: pd.DataFrame, return_type: str = "arithmetic") -> pd.DataFrame:
        """
        Compute the returns of the input data.

        Parameters:
            data (pd.DataFrame): The input data to compute returns.
            return_type (str): The type of returns to compute ("arithmetic" or "logarithmic").

        Returns:
            pd.DataFrame: The computed returns.
        """
        if return_type == "logarithmic":
            returns = np.log(data / data.shift(1))
        else:  # arithmetic
            returns = data.pct_change()
        return returns

    @staticmethod
    def plot_comparison(data: pd.DataFrame, transformed_data: pd.DataFrame, returns: pd.DataFrame, n: int) -> None:
        """
        Plot the original, fractionally differentiated, and returns data for comparison.

        Parameters:
            data (pd.DataFrame): The original input data.
            transformed_data (pd.DataFrame): The fractionally differentiated data.
            returns (pd.DataFrame): The computed returns data.
            n (int): The number of columns to plot.
        """
        columns_to_plot = data.columns[:n]
        num_plots = len(columns_to_plot)

        fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(14, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]

        for i, column in enumerate(columns_to_plot):
            axes[i].plot(data.index, data[column], label='Original', color='blue')
            axes[i].plot(transformed_data.index, transformed_data[column], label='Transformed', color='orange')
            axes[i].plot(returns.index, returns[column], label='Returns', color='green')
            axes[i].set_title(f'Data - {column}')
            axes[i].legend(loc='best')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from ml_returns_pred.file_management.file_manager import FileManagerStatic
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor

    fms = FileManagerStatic()

    relative_data_path = "../../data/raw_data/canadian_stocks_2000-01-01_to_2024-06-01.csv"
    prices_data = fms.load_data(relative_file_path=relative_data_path, index_col=0).iloc[1:, :]
    print(prices_data.head(8))

    dp = DataPreprocessor()
    prices_data_preprocessed = dp.preprocess(data=prices_data)

    fd = FractionalDifferentiation()

    transformed_data = fd.fracdiff(data=prices_data_preprocessed, d=0.5)
    returns = fd.compute_returns(data=prices_data_preprocessed, return_type="logarithmic").iloc[1:, :]

    fd.plot_comparison(data=prices_data_preprocessed, transformed_data=transformed_data, returns=returns, n=3)

    transformed_data_stat = fd.fracdiff_stat(data=prices_data_preprocessed)
    fd.plot_comparison(data=prices_data_preprocessed, transformed_data=transformed_data_stat, returns=returns, n=3)





