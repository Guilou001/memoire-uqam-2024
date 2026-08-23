import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCABasedVariableSelector:
    """
    A class to perform PCA-based variable selection and transformation on time series data.

    Attributes
    ----------
    data : pd.DataFrame
        The input time series data.
    n_components : int
        The number of principal components to keep.
    scaler : StandardScaler
        Scaler to normalize the data.
    pca : PCA
        PCA model to fit the data.
    """

    def __init__(self, n_components: int):
        """
        Initialize the PCABasedVariableSelector with the number of principal components.

        Parameters
        ----------
        n_components : int
            The number of principal components to keep.
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.important_features = None
        self.data = None

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with NaN values.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data without NaN values.
        """
        if data.isnull().sum().sum() > 0:
            print(f"Warning: Dropping columns {data.columns[data.isna().any()].tolist()} with NaN values.")

        return data.dropna(axis=1)

    def fit(self, data: pd.DataFrame):
        """
        Fit the PCA model to the data and identify the most important features.

        Parameters
        ----------
        data : pd.DataFrame
            The input time series data.
        """

        self.data = self._prepare_data(data=data)

        scaled_data = self._scale_data(self.data)
        self.pca.fit(X=scaled_data)
        self.important_features = self._get_important_features_indices()

    def _scale_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scale the data using StandardScaler.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        np.ndarray
            The scaled data.
        """
        return self.scaler.fit_transform(X=data)

    def _get_important_features_indices(self) -> np.ndarray:
        """
        Get indices of the most important features based on PCA loadings.

        Returns
        -------
        np.ndarray
            Indices of the most important features.
        """
        loadings = np.abs(self.pca.components_)
        return np.argsort(-loadings, axis=1)[:, :self.n_components]

    def transform(self) -> pd.DataFrame:
        """
        Transform the data to keep only the most important features.

        Returns
        -------
        pd.DataFrame
            The transformed data with the most important features.
        """
        if self.important_features is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before transform().")

        selected_features = np.unique(self.important_features.flatten())
        return self.data.iloc[:, selected_features]

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the PCA model and transform the data to keep only the most important features.

        Parameters
        ----------
        data : pd.DataFrame
            The input time series data.

        Returns
        -------
        pd.DataFrame
            The transformed data with the most important features.
        """
        self.fit(data)
        return self.transform()

    def get_important_features(self) -> list:
        """
        Get the list of important features selected by PCA.

        Returns
        -------
        list
            The list of important features.
        """
        if self.important_features is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before get_important_features().")

        return np.unique(self.important_features.flatten()).tolist()

    def plot_explained_variance(self, save_plot_path: str = None):
        """
        Plot the cumulative explained variance by the number of principal components.

        Parameters
        ----------
        save_plot_path : str, optional
            The path to save the plot. If None, the plot is displayed without saving.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)

        if save_plot_path:
            plt.savefig(f"{save_plot_path}variance_explained_by_components.png")
            plt.close()
        else:
            plt.show()

    def get_loadings(self) -> pd.DataFrame:
        """
        Get the loadings of the original variables on the principal components.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loadings of the original variables on the principal components.
        """
        if self.important_features is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before get_loadings().")

        loadings = pd.DataFrame(self.pca.components_.T, index=self.data.columns,
                                columns=[f'PC{i + 1}' for i in range(self.n_components)])
        return loadings

    def get_principal_components(self) -> pd.DataFrame:
        """
        Transform the original data to its principal components.

        Returns
        -------
        pd.DataFrame
            The transformed data with principal components.
        """
        if self.data is None:
            raise ValueError("The model has not been fitted yet. Please call fit() before get_principal_components().")

        scaled_data = self.scaler.transform(self.data)
        principal_components = self.pca.transform(scaled_data)
        return pd.DataFrame(principal_components, index=self.data.index,
                            columns=[f'PC{i+1}' for i in range(self.n_components)])


if __name__ == '__main__':
    from ml_returns_pred.file_management.file_manager import FileManagerDynamic

    # ----------------------- LOADING data -----------------------#

    fms = FileManagerDynamic(ceiling_directory="MSC")
    macro_data = fms.load_data(folder_name="raw_data", file_name="macro_data.csv", index_col=0)
    macro_data.index = pd.to_datetime(macro_data.index)

    print(macro_data.head(8))
    print(macro_data.shape)

    # drop columns where nan are presents
    macro_data = macro_data.dropna(axis=1)
    print(macro_data.shape)

    pca_selector = PCABasedVariableSelector(n_components=100)
    transformed_data = pca_selector.fit_transform(data=macro_data)
    important_features = pca_selector.get_important_features()

    print(f"Important features: {important_features}")
    print(f"Transformed data shape: {transformed_data.shape}")
    print(transformed_data.head())

    # Plot explained variance
    pca_selector.plot_explained_variance()

    # Get principal components
    principal_components = pca_selector.get_principal_components()
    print(f"Principal Components shape: {principal_components.shape}")
    print(principal_components.head())
