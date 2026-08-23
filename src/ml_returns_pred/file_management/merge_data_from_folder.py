import os

import pandas as pd

from ml_returns_pred.file_management.file_manager import FileManagerStatic


class MergeDataFromFolder:
    """
    A class to manage and concatenate all DataFrames from a given folder.

    Attributes:
        folder_path (str): The path to the folder containing the CSV files.
        file_extension (str): The extension of the files to be read (default is '.csv').
    """

    DEFAULT_FILE_EXTENSION = '.csv'
    DEFAULT_RENAME_STRATEGY = 'replace'
    DEFAULT_SAVE_PATH = '../../data/concatenated_data/compute_strategy_returns/strategy_returns_concat.csv'

    def __init__(self, folder_path: str, file_extension: str = DEFAULT_FILE_EXTENSION):
        """
        Initializes the MergeDataFromFolder instance with a folder path and file extension.

        Args:
            folder_path (str): The path to the folder containing the CSV files.
            file_extension (str): The extension of the files to be read (default is '.csv').
        """
        self.folder_path = folder_path
        self.file_extension = file_extension

    def get_file_list(self) -> list[str]:
        """
        Returns a list of file paths in the folder.

        Returns:
            List[str]: A list of file paths in the folder with the specified extension.
        """
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(self.file_extension)
        ]

    @staticmethod
    def read_dataframe(file_path: str, **read_kwargs) -> pd.DataFrame:
        """
        Reads a single file and returns a DataFrame.

        Args:
            file_path (str): The file path to read.
            **read_kwargs: Additional arguments for pd.read_csv.

        Returns:
            pd.DataFrame: The DataFrame read from the file.
        """
        return FileManagerStatic().load_data(relative_file_path=file_path, **read_kwargs)

    def read_dataframes(self, file_list: list[str], **read_kwargs) -> dict[str, pd.DataFrame]:
        """
        Reads the files and returns a dictionary of DataFrames.

        Args:
            file_list (List[str]): A list of file paths to read.
            **read_kwargs: Additional arguments for pd.read_csv.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with file names as keys and DataFrames as values.
        """
        return {
            os.path.splitext(os.path.basename(file))[0]: self.read_dataframe(file, **read_kwargs)
            for file in file_list
        }

    def merge_dataframes(self, dataframes: dict[str, pd.DataFrame], rename_strategy: str = DEFAULT_RENAME_STRATEGY,
                         **merge_kwargs) -> pd.DataFrame:
        """
        Merges the dictionary of DataFrames and distinguishes column names by adding the file name.

        Args:
            dataframes (Dict[str, pd.DataFrame]): A dictionary with file names as keys and DataFrames as values.
            rename_strategy (str): The strategy for renaming columns: 'suffix' to add a suffix, 'replace' to replace column names with file names.
            **merge_kwargs: Additional arguments for pd.merge.

        Returns:
            pd.DataFrame: The merged DataFrame with distinguished column names.
        """
        merged_df = None
        for name, df in dataframes.items():
            df_renamed = self._rename_columns(df, name, rename_strategy)
            if merged_df is None:
                merged_df = df_renamed
            else:
                merged_df = merged_df.merge(df_renamed, left_index=True, right_index=True, **merge_kwargs)
        return merged_df

    @staticmethod
    def _rename_columns(df: pd.DataFrame, name: str, strategy: str) -> pd.DataFrame:
        """
        Renames the columns of a DataFrame based on the specified strategy.

        Args:
            df (pd.DataFrame): The DataFrame whose columns need to be renamed.
            name (str): The name to use for renaming.
            strategy (str): The strategy for renaming columns: 'suffix' to add a suffix, 'replace' to replace column names with file names.

        Returns:
            pd.DataFrame: The DataFrame with renamed columns.
        """
        if strategy == 'suffix':
            return df.add_suffix(f"_{name}")
        elif strategy == 'replace':
            return df.rename(columns=lambda col: f"{name}")
        else:
            raise ValueError("rename_strategy must be either 'suffix' or 'replace'")

    @staticmethod
    def save_dataframe(data: pd.DataFrame, save_path: str, save_kwargs: dict | None = None):
        """
        Saves the data to a CSV file.

        Args:
            data (pd.DataFrame): The data to save.
            save_path (str): The path to save the data.
            save_kwargs (dict, optional): Additional arguments for pd.DataFrame.to_csv.
        """
        if save_kwargs is None:
            save_kwargs = {}
        FileManagerStatic().save_data(data=data, relative_file_path=save_path, **save_kwargs)

    def merge(self, read_kwargs: dict | None = None, merge_kwargs: dict | None = None,
              rename_strategy: str = DEFAULT_RENAME_STRATEGY, save_data: bool = True,
              save_path: str = DEFAULT_SAVE_PATH, save_kwargs: dict | None = None) -> pd.DataFrame:
        """
        The main method to merge all DataFrames from the folder.

        Args:
            read_kwargs (dict, optional): Additional arguments for pd.read_csv.
            merge_kwargs (dict, optional): Additional arguments for pd.merge.
            rename_strategy (str, optional): The strategy for renaming columns: 'suffix' to add a suffix,
            'replace' to replace column names with file names.
            save_data (bool, optional): Whether to save the merged DataFrame to a CSV file.
            save_path (str, optional): The path to save the merged DataFrame.
            save_kwargs (dict, optional): Additional arguments for pd.DataFrame.to_csv.

        Returns:
            pd.DataFrame: The merged DataFrame from all files in the folder.
        """
        read_kwargs = read_kwargs or {}
        merge_kwargs = merge_kwargs or {}
        save_kwargs = save_kwargs or {}

        file_list = self.get_file_list()
        dataframes = self.read_dataframes(file_list, **read_kwargs)
        merged_dataframes = self.merge_dataframes(dataframes, rename_strategy, **merge_kwargs)

        if save_data:
            self.save_dataframe(merged_dataframes, save_path, save_kwargs)

        return merged_dataframes


class BatchMergeDataFromFolders:
    """
    A class to manage and concatenate DataFrames from multiple folders, saving the results to specified paths.

    Attributes
    ----------
    folder_paths : List[str]
        List of paths to the folders containing the CSV files.
    save_paths : List[str]
        List of paths to save the concatenated DataFrames.
    file_extension : str
        The extension of the files to be read (default is '.csv').
    """

    def __init__(self, folder_paths: list[str], save_paths: list[str], file_extension: str = '.csv'):
        """
        Initializes the BatchMergeDataFromFolders instance with folder paths, save paths, and file extension.

        Parameters
        ----------
        folder_paths : List[str]
            List of paths to the folders containing the CSV files.
        save_paths : List[str]
            List of paths to save the concatenated DataFrames.
        file_extension : str, optional
            The extension of the files to be read (default is '.csv').
        """
        self.folder_paths = folder_paths
        self.save_paths = save_paths
        self.file_extension = file_extension

    def batch_merge(self, read_kwargs: dict | None = None, merge_kwargs: dict | None = None,
                    rename_strategy: str = MergeDataFromFolder.DEFAULT_RENAME_STRATEGY,
                    save_kwargs: dict | None = None, save_data: bool = True) -> list[pd.DataFrame]:
        """
        Merges DataFrames from multiple folders and saves the results to the specified paths.

        Parameters
        ----------
        read_kwargs : dict, optional
            Additional arguments for pd.read_csv.
        merge_kwargs : dict, optional
            Additional arguments for pd.merge.
        rename_strategy : str, optional
            The strategy for renaming columns: 'suffix' to add a suffix,
            'replace' to replace column names with file names (default is 'replace').
        save_kwargs : dict, optional
            Additional arguments for pd.DataFrame.to_csv.
        save_data : bool, optional

        Returns
        -------
        List[pd.DataFrame]
            List of merged DataFrames from all specified folders.
        """
        merged_dataframes = []
        for folder_path, save_path in zip(self.folder_paths, self.save_paths):
            concatenator = MergeDataFromFolder(folder_path=folder_path, file_extension=self.file_extension)
            merged_df = concatenator.merge(read_kwargs=read_kwargs, merge_kwargs=merge_kwargs,
                                           rename_strategy=rename_strategy, save_data=save_data,
                                           save_path=save_path, save_kwargs=save_kwargs)
            merged_dataframes.append(merged_df)
        return merged_dataframes


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    folder_paths = [
        "../../data/intermediate_data/compute_strategy_returns/",
        "../../data/intermediate_data/evaluate_model_performance/"
    ]
    save_paths = [
        "../../data/concatenated_data/compute_strategy_returns/strategy_returns_concat.csv",
        "../../data/concatenated_data/evaluate_model_performance/model_performance_metrics_concat.csv"
    ]
    read_kwargs = {
        'index_col': 0
    }
    merge_kwargs = {
        'how': 'outer'
    }
    rename_strategy = 'replace'

    merger = BatchMergeDataFromFolders(folder_paths=folder_paths, save_paths=save_paths)
    merger.batch_merge(read_kwargs=read_kwargs, merge_kwargs=merge_kwargs, rename_strategy=rename_strategy)

    print("DataFrames merged and saved successfully!")
