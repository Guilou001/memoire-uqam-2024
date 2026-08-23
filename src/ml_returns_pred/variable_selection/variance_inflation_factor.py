
import numpy as np
import pandas as pd
from numba import float64, jit


@jit(float64[:](float64[:, :]), nopython=True)
def calculate_vif_numba(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the data matrix.

    Parameters:
    - feature_matrix (np.ndarray): The feature matrix.

    Returns:
    - np.ndarray: An array containing the VIF values for each feature.
    """
    num_features = feature_matrix.shape[1]
    vif_values = np.zeros(num_features)

    for feature_index in range(num_features):
        target_feature = np.ascontiguousarray(feature_matrix[:, feature_index])
        excluded_indices = np.arange(num_features) != feature_index
        predictor_features = np.ascontiguousarray(feature_matrix[:, excluded_indices])

        # Calculate R-squared for the auxiliary OLS regression using matrix inversion
        regression_coefficients = (np.linalg.inv(predictor_features.T @ predictor_features)
                                   @ predictor_features.T @ target_feature)
        predicted_values = predictor_features @ regression_coefficients
        residuals = target_feature - predicted_values
        residual_sum_of_squares = np.sum(residuals ** 2)
        total_sum_of_squares = np.sum((target_feature - np.mean(target_feature)) ** 2)
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

        # Calculate VIF
        vif_values[feature_index] = 1. / (1. - r_squared)

    return vif_values


class VIFSelector:
    """
    An optimized feature selector that removes variables with the highest Variance Inflation Factor (VIF)
    until a specified number of variables are left.
    """

    def __init__(self, n_features: int):
        """
        Initialize the VIFSelector object.

        Parameters:
        - n_features (int): The number of features to select.
        """
        self.n_features = n_features
        self.selected_features: list[str] = []

    def fit(self, X: pd.DataFrame, verbose: bool = False) -> 'VIFSelector':
        """
        Fit the feature selector to the data.

        Parameters:
        - X (pd.DataFrame): The feature matrix.
        - verbose (bool): Flag to indicate if verbose output should be printed.

        Returns:
        - self: The fitted object.
        """
        # Drop columns with NaN values for the VIF computation
        X_dropped_na = X.dropna(axis=1)
        X_matrix = X_dropped_na.values

        # Initialize variables
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_dropped_na.columns

        while len(vif_data) > self.n_features:
            # Calculate VIF for the remaining features
            vif_values = calculate_vif_numba(X_matrix)
            vif_data["VIF"] = vif_values

            # Find the variable with the highest VIF
            remove = vif_data.sort_values("VIF", ascending=False).iloc[0]

            if verbose:
                print(f"{len(vif_data)} - Removing {remove['feature']} with VIF {remove['VIF']:.4f}")

            # Drop the variable with the highest VIF from the dataset
            X_dropped_na = X_dropped_na.drop(remove["feature"], axis=1)
            X_matrix = X_dropped_na.values
            vif_data = vif_data[vif_data["feature"] != remove["feature"]]

        self.selected_features = vif_data["feature"].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature matrix by keeping only the selected features.

        Parameters:
        - X (pd.DataFrame): The feature matrix.

        Returns:
        - pd.DataFrame: The transformed feature matrix.
        """
        return X[self.selected_features]


class VIFSelectorManualCategories:
    def __init__(self):
        self.selected_features_per_category = {}

    def fit(self, X: pd.DataFrame, factor_categories: dict, verbose: bool = False) -> 'VIFSelectorManualCategories':
        """
        Fit the feature selector to the data.

        Parameters:
        - X (pd.DataFrame): The feature matrix.
        - factor_categories (dict): A dictionary containing the factor categories and their factors.
        - verbose (bool): A flag to indicate if verbose output should be printed.

        Returns:
        - self: The fitted object.
        """
        # Drop columns with NaN values for the VIF computation
        X_dropped_na = X.dropna(axis=1)

        for category, factors in factor_categories.items():
            common_factors = set(factors).intersection(set(X_dropped_na.columns))
            if common_factors:
                subset_X = X_dropped_na[list(common_factors)]
                X_matrix = subset_X.values

                # Calculate VIF for the factors_returns in this category
                vif_values = calculate_vif_numba(X_matrix)
                vif_data = pd.DataFrame({"feature": list(common_factors), "VIF": vif_values})

                # Find the variable with the highest VIF
                max_vif_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]['feature']
                self.selected_features_per_category[category] = max_vif_feature

                if verbose:
                    print(
                        f"Category: {category}, Selected Feature: {max_vif_feature} "
                        f"with VIF: {vif_data.loc[vif_data['feature'] == max_vif_feature, 'VIF'].values[0]:.4f}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected_features = list(self.selected_features_per_category.values())
        return X[selected_features]


if __name__ == '__main__':
    print("This is variable_selection.py")
    from ml_returns_pred.file_management.file_manager import FileManagerDynamic

    #----------------------- LOADING data -----------------------#

    fms = FileManagerDynamic(ceiling_directory="MSC")
    macro_data = fms.load_data(folder_name="raw_data", file_name="macro_data.csv", index_col=0)

    print(macro_data.head(8))
    print(macro_data.shape)

    # drop columns where nan are presents

    macro_data = macro_data.dropna(axis=1)
    print(macro_data.shape)

    vif_selector = VIFSelector(n_features=30)
    vif_selector.fit(X=macro_data, verbose=True)
    print(vif_selector.selected_features)

    '''
    ['hstart_NF_new', 'hstart_NS_new', 'hstart_NB_new', 'hstart_ONT_new', 'hstart_MAN_new', 'hstart_ALB_new', 
    'build_Comm_NB_new', 'build_Ind_QC_new', 'build_Ind_ONT_new', 'build_Comm_MAN_new', 'build_Comm_BC_new', 
    'N_DUR_INV_RAT_new', 'CRED_HOUS_non_MORT', 'CRED_BUS', 'TBILL_6M.Bank_rate', 'GBPCAD_new', 'IPPI_WOOD_CAN', 
    'TSX_CLO', 'EMP_FIN_PEI', 'EMP_FIN_NS', 'EMP_MANU_NS', 'EMP_FOR_OIL_ONT', 'EMP_FIN_ONT', 'EMP_CONS_MAN', 
    'EMP_FIN_BC', 'UNEMP_DURAvg_MAN_new', 'UNEMP_DURAvg_ALB_new', 'CLAIMS_PEI', 'CPI_CLOT_NF', 'CPI_DUR_NS']
    '''









