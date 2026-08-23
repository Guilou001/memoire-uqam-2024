import numpy as np
import pandas as pd

from ml_returns_pred.fractional_differentiation.fractional_differentiation import FractionalDifferentiation


class FromPricesToReturns:

    @staticmethod
    def _verify_return_type(return_type: str) -> str:
        """
        Verifies if the provided return type is valid.

        Parameters
        ----------
        return_type : str
            The return type to be verified.

        Returns
        -------
        str
            The verified return type.
        """

        valid_return_types = ['arithmetic', 'logarithmic']

        if return_type.lower() not in valid_return_types:
            available_return_types = ", ".join(f"'{key}'" for key in valid_return_types)
            raise ValueError(
                f"Invalid return type: '{return_type}'. Available return types are: {available_return_types}."
            )

        return return_type

    def compute_returns(self, data: pd.DataFrame | pd.Series, return_type: str = 'arithmetic', binarize: bool = False,
                        fractional_differentiation: bool = False) -> pd.DataFrame:
        """
        Compute the returns of the price data.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            The price data to compute returns from.
        return_type : str
            The method to compute returns ('arithmetic' or 'logarithmic').
        binarize : bool
            Whether to binarize the returns.
        fractional_differentiation : bool
            Whether to apply fractional differentiation.

        Returns
        -------
        pd.DataFrame
            The returns of the price data.
        """

        if fractional_differentiation:
            frac_diff = FractionalDifferentiation()
            frac_diff_data = frac_diff.fracdiff_stat(data=data)
            return frac_diff_data[1:]
        else:
            return_type = self._verify_return_type(return_type=return_type)

            if return_type == 'arithmetic':
                returns = data.pct_change()
            else:
                returns = np.log(data).diff()

            if binarize:
                returns = self.binarize_returns(returns=returns)

            return returns[1:]

    @staticmethod
    def binarize_returns(returns: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
        """
        Binarize the returns of the price data.

        Parameters
        ----------
        returns : pd.DataFrame
            The returns of the price data.
        threshold : float
            The threshold to binarize the returns.

        Returns
        -------
        pd.DataFrame
            The binarized returns of the price data.
        """
        binarized_returns = returns.map(lambda x: 1 if x > threshold else 0).astype(int)

        return binarized_returns


if __name__ == '__main__':
    from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
    from ml_returns_pred.read_data.data_reader import DataReader
    from ml_returns_pred.resample_data.data_resampler import DataResampler

    dr = DataReader()
    relative_data_path = "../../data/raw_data/canadian_stocks_data.csv"
    relative_macro_data_path = "../../data/raw_data/macro_data.csv"
    prices_data = dr.read_single_columns_level_data(relative_file_path=relative_data_path, index_col=0)
    macro_data = dr.read_single_columns_level_data(relative_file_path=relative_macro_data_path, index_col=0)
    print(prices_data.head(8))

    dp = DataPreprocessor()
    prices_data_preprocessed = dp.preprocess(data=prices_data)
    macro_data_preprocessed = dp.preprocess(data=macro_data)

    prices_data_aligned, macro_data_aligned = dp.align_dataframes_within_common_period(
        dataframe_1=prices_data_preprocessed,
        dataframe_2=macro_data_preprocessed
    )

    dr = DataResampler()

    prices_data_resampled = dr.resample_and_forward_fill(
        data_to_resample=prices_data_aligned,
        reference_data=macro_data_aligned
    )

    print(prices_data_resampled.head(8))

    fptr = FromPricesToReturns()
    returns = fptr.compute_returns(data=prices_data_resampled, return_type='logarithmic')
    print(returns.head())
