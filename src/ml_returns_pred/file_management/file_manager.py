import importlib.util
import os
import shutil
from typing import Any, Union

import pandas as pd


class FileManagerStatic:
    """
    A static class to manage file operations with relative paths.
    """

    def __init__(self, base_directory: str = os.getcwd()):
        """
        Initialize FileManagerStatic class with optional base directory relative to the current working directory.
        """
        self.base_directory = base_directory

    def _get_full_path(self, relative_path: str) -> str:
        """
        Construct the full path from the base directory and the relative path.

        Parameters:
        -----------
        relative_path : str
            The relative path from the base directory.

        Returns:
        --------
        str
            The full path constructed from the base directory and the relative path.
        """
        return os.path.join(self.base_directory, relative_path)

    def load_data(self, relative_file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a specified relative file path.

        Parameters:
        -----------
        relative_file_path : str
            The relative path to the file to read.
        **kwargs : dict
            Additional keyword arguments to pass to the pandas reading function.

        Returns:
        --------
        pd.DataFrame
            The data read from the file.
        """
        full_file_path = self._get_full_path(relative_file_path)
        file_extension = os.path.splitext(full_file_path)[1]

        if file_extension == '.csv':
            return pd.read_csv(full_file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(full_file_path, **kwargs)
        elif file_extension in ['.parquet']:
            return pd.read_parquet(full_file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def save_data(self, relative_file_path: str, data: pd.DataFrame, **kwargs) -> None:
        """
        Save data to a specified relative file path.

        Parameters:
        -----------
        relative_file_path : str
            The relative path where the file will be saved.
        data : pd.DataFrame
            The data to save.
        **kwargs : dict
            Additional keyword arguments to pass to the pandas writing function.
        """
        full_file_path = self._get_full_path(relative_file_path)
        file_extension = os.path.splitext(full_file_path)[1]

        if file_extension == '.csv':
            data.to_csv(full_file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            data.to_excel(full_file_path, **kwargs)
        elif file_extension in ['.parquet']:
            data.to_parquet(full_file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")


class FileManagerDynamic:
    """
    A class to manage file operations including loading and saving data.
    """

    def __init__(self, ceiling_directory: str = None):
        """
        Initialize FileManagerDynamic class with optional ceiling directory.
        If ceiling directory is None, initialize it by stepping back n subdirectories.
        """
        if ceiling_directory is None:
            self.set_ceiling_directory()
        else:
            self.ceiling_directory = ceiling_directory

    def set_ceiling_directory(self, steps_back: int = 3, verbose: bool = False):
        """
        Set the ceiling directory by stepping back a given number of subdirectories.

        Parameters:
        -----------
        steps_back : int
            Number of subdirectories to step back to set the ceiling directory.
        """
        path = os.getcwd()
        for _ in range(steps_back):
            path = os.path.dirname(path)
        self.ceiling_directory = path
        if verbose:
            print(f"Ceiling directory set to: {self.ceiling_directory}")

    def search(self, target_name: str, start_path: str, search_type: str = 'both') -> Union[str, None]:
        """
        Search for a target starting from a given path using DFS.

        Parameters
        ----------
        target_name : str
            The name of the target (file or folder) to search for.
        start_path : str
            The path from where to start the search.
        search_type : str, optional
            The type of element to search for ('file', 'folder', or 'both').

        Returns
        -------
        str or None
            The path where the target was found, or None if not found.
        """

        visited = set()

        def dfs_search(current_path: str) -> Union[str, None]:
            # Check if the current path has already been visited or if it's the ceiling directory
            if current_path in visited or (self.ceiling_directory and current_path.endswith(self.ceiling_directory)):
                return None

            # Mark the current path as visited
            visited.add(current_path)

            # Get a list of all names (files and folders) in the current directory
            all_names = os.listdir(current_path)

            # Check if the target name exists in the current directory
            if target_name in all_names:
                # Construct the full path to the target
                full_path = os.path.join(current_path, target_name)

                # Validate against the search type ('file', 'folder', or 'both')
                if (search_type == 'both' or
                        (search_type == 'file' and os.path.isfile(full_path)) or
                        (search_type == 'folder' and os.path.isdir(full_path))):
                    return full_path  # Target found

            # If target not found, search in sub-folders
            for name in all_names:
                new_path = os.path.join(current_path, name)
                # If it's a directory, perform a DFS on it
                if os.path.isdir(new_path):
                    result = dfs_search(new_path)
                    if result:
                        return result  # Target found in one of the sub-folders

            # If target is still not found, move up one directory and perform DFS
            parent_path = os.path.dirname(current_path)
            if parent_path and parent_path != current_path:
                return dfs_search(parent_path)

            return None  # Target not found

        return dfs_search(start_path)

    def load_data(self, folder_name: str, file_name: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a specified folder and file.

        Parameters:
        -----------
        folder_name : str
            The name of the folder containing the file.
        file_name : str
            The name of the file to read.
        **kwargs : dict
            Additional keyword arguments to pass to the pandas reading function.

        Returns:
        --------
        pd.DataFrame
            The data read from the file.
        """
        # Start from the current working directory
        start_path = os.getcwd()

        # Search for the target folder
        folder_path = self.search(target_name=folder_name, start_path=start_path, search_type='folder')
        if folder_path is None:
            raise FileNotFoundError(f"Folder '{folder_name}' not found.")

        # Construct the complete file path
        file_path = os.path.join(folder_path, file_name)

        # Detect the file extension
        file_extension = os.path.splitext(file_name)[1]

        # Use the appropriate pandas function to read the file based on its extension
        if file_extension == '.csv':
            data = pd.read_csv(file_path, **kwargs)
        elif file_extension in ['.parquet']:
            data = pd.read_parquet(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        print(f"Loaded file '{file_name}' from folder '{folder_name}'")

        return data

    def save_data(self, folder_name: str, file_name: str, data: pd.DataFrame, verbose: bool = True, **kwargs) -> None:
        """
        Save data to a specified folder and file.

        Parameters:
        -----------
        folder_name : str
            The name of the folder where the file will be saved.
        file_name : str
            The name of the file to save.
        data : pd.DataFrame
            The data to save.
        **kwargs : dict
            Additional keyword arguments to pass to the pandas writing function.
        """
        # Start from the current working directory
        start_path = os.getcwd()

        # Search for the target folder
        folder_path = self.search(target_name=folder_name, start_path=start_path, search_type='folder')
        if folder_path is None:
            raise FileNotFoundError(f"Folder '{folder_name}' not found.")

        # Construct the complete file path
        file_path = os.path.join(folder_path, file_name)

        # Detect the file extension
        file_extension = os.path.splitext(file_name)[1]

        # Use the appropriate pandas function to save the file based on its extension
        if file_extension == '.csv':
            data.to_csv(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            data.to_excel(file_path, **kwargs)
        elif file_extension in ['.parquet']:
            data.to_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        if verbose:
            print(f"Saved file '{file_name}' to folder '{folder_name}'")

    def file_exists(self, folder_name: str, file_name: str) -> bool:
        """
        Check if a given file exists in a specified folder.

        Parameters:
        -----------
        folder_name : str
            The name of the folder to check.
        file_name : str
            The name of the file to check.

        Returns:
        --------
        bool
            True if the file exists, False otherwise.
        """
        folder_path = self.search(target_name=folder_name, start_path=os.getcwd(), search_type='folder')
        if folder_path is None:
            return False
        return file_name in os.listdir(folder_path)

    def list_files(self, folder_name: str, extension: str = None) -> list:
        """
        List all files of a given type in a specified folder.

        Parameters:
        -----------
        folder_name : str
            The name of the folder to check.
        extension : str, optional
            The file extension to filter by.

        Returns:
        --------
        list
            A list of file names that match the criteria.
        """
        folder_path = self.search(target_name=folder_name, start_path=os.getcwd(), search_type='folder')
        if folder_path is None:
            return []
        files = os.listdir(folder_path)
        if extension:
            return [f for f in files if f.endswith(extension)]
        return files

    def delete_file(self, folder_name: str, file_name: str) -> None:
        """
        Delete a specified file from a specified folder.

        Parameters:
        -----------
        folder_name : str
            The name of the folder containing the file.
        file_name : str
            The name of the file to delete.
        """
        folder_path = self.search(target_name=folder_name, start_path=os.getcwd(), search_type='folder')
        if folder_path is None:
            raise FileNotFoundError(f"File '{file_name}' in folder '{folder_name}' not found.")
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)

    def rename_file(self, folder_name: str, old_file_name: str, new_file_name: str) -> None:
        """
        Rename a specified file in a specified folder.

        Parameters:
        -----------
        folder_name : str
            The name of the folder containing the file.
        old_file_name : str
            The current name of the file.
        new_file_name : str
            The new name for the file.
        """
        folder_path = self.search(target_name=folder_name, start_path=os.getcwd(), search_type='folder')
        if folder_path is None:
            raise FileNotFoundError(f"File '{old_file_name}' in folder '{folder_name}' not found.")
        old_file_path = os.path.join(folder_path, old_file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)

        print(f"Renamed file '{old_file_name}' to '{new_file_name}'")

    def move_file(self, src_folder: str, dest_folder: str, file_name: str) -> None:
        """
        Move a specified file from one folder to another.

        Parameters:
        -----------
        src_folder : str
            The name of the source folder.
        dest_folder : str
            The name of the destination folder.
        file_name : str
            The name of the file to move.
        """
        # Special case: if src_folder is the current directory or None
        if src_folder == os.getcwd() or not src_folder:
            src_folder_path = os.getcwd()
        else:
            src_folder_path = self.search(target_name=src_folder, start_path=os.getcwd(), search_type='folder')
        dest_folder_path = self.search(target_name=dest_folder, start_path=os.getcwd(), search_type='folder')

        # Check if the source and destination folders were found
        if src_folder_path is None or dest_folder_path is None:
            raise FileNotFoundError(f"Source folder '{src_folder}' or destination folder '{dest_folder}' not found.")

        src_file_path = os.path.join(src_folder_path, file_name)
        dest_file_path = os.path.join(dest_folder_path, file_name)

        # Check if the source file exists before attempting to move it
        if not os.path.exists(src_file_path):
            raise FileNotFoundError(f"Source file '{file_name}' not found in folder '{src_folder}'.")

        shutil.move(src_file_path, dest_file_path)
        print(f"Moved file '{file_name}' from '{src_folder}' to '{dest_folder}'")

    def import_local_package(self, module_name: str, start_path: str = None) -> object:
        """
        Import a Python module from a local path dynamically.

        Parameters:
        -----------
        module_name : str
            The name of the module to import.
        start_path : str, optional
            The path from where to start the search for the module. Defaults to the current working directory.

        Returns:
        --------
        object
            The imported module object.
        """

        if start_path is None:
            start_path = os.getcwd()

        # Search for the target module
        module_path = self.search(f'{module_name}.py', start_path=start_path, search_type='file')
        if module_path is None:
            raise ImportError(f"Module '{module_name}' not found.")

        # Dynamic import using importlib
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def import_class(self, module_name: str, class_or_variable_name: str, start_path: str = None) -> Any:
        """
        Import a specific class or variable from a local module.

        Parameters:
        -----------
        module_name : str
            The name of the module from which to import.
        class_or_variable_name : str
            The name of the class or variable to import.
        start_path : str, optional
            The path from which to start searching for the module. Defaults to the current working directory.

        Returns:
        --------
        Any
            The imported class or variable.
        """

        # Import the module using the existing method
        module = self.import_local_package(module_name, start_path)

        # Get the class or variable using getattr
        class_or_variable = getattr(module, class_or_variable_name, None)

        if class_or_variable is None:
            raise ImportError(f"Class or variable '{class_or_variable_name}' not found in module '{module_name}'.")

        return class_or_variable


if __name__ == '__main__':
    print("This is the file management file")



    # Pour la version statique
    # time_static = timeit.timeit(
    #     "fm_static.load_data(relative_file_path=f'{raw_data_folder_name_static}/SG_Long vs Short_TSX.xlsx', index_col=0)",
    #     globals=globals(),
    #     number=10
    # )
    #
    # # Pour la version dynamique
    # time_dynamic = timeit.timeit(
    #     "fm.load_data(folder_name=raw_data_folder_name_dynamic, file_name='SG_Long vs Short_TSX.xlsx', index_col=0)",
    #     globals=globals(),
    #     number=10
    # )
    #
    # print(f"FileManagerStatic load_data time: {time_static / 10} seconds per loop")
    # print(f"FileManagerDynamic load_data time: {time_dynamic / 10} seconds per loop")