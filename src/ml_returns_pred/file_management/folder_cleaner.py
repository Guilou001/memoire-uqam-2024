import os


class FolderCleaner:
    def __init__(self, folder_paths: list[str], extensions: list[str] | None = None) -> None:
        """
        Initialize the FolderCleaner with a list of folder paths and optional file extensions to clean.

        Parameters
        ----------
        folder_paths : List[str]
            List of relative paths to folders that need to be cleaned.
        extensions : Optional[List[str]]
            List of file extensions to clean. Only files with these extensions will be deleted.
            If None, all files will be deleted. Example: ['.csv', '.png']
        """
        self.folder_paths = folder_paths
        self.extensions = extensions

    def clean_folders(self) -> None:
        """
        Clean each folder specified in the folder_paths list by deleting files with the specified extensions.
        """
        for folder_path in self.folder_paths:
            full_path = os.path.abspath(folder_path)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                self._clean_folder(full_path)
                print(f"Cleaned folder: {full_path}")
            else:
                print(f"Folder does not exist or is not a directory: {full_path}")

    def _clean_folder(self, folder_path: str) -> None:
        """
        Helper method to delete files in a folder with specified extensions.

        Parameters
        ----------
        folder_path : str
            The absolute path to the folder to clean.
        """
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                if not self.extensions or any(filename.endswith(ext) for ext in self.extensions):
                    self._delete_file(file_path)

    @staticmethod
    def _delete_file(file_path: str) -> None:
        """
        Helper method to delete a file and handle exceptions.

        Parameters
        ----------
        file_path : str
            The absolute path to the file to delete.
        """
        try:
            os.unlink(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == '__main__':
    # Example usage
    folders_to_clean = ["../../data/intermediate_data/"]
    extensions_to_clean = ['.csv', '.png']  # Specify extensions or set to None to clean all files
    cleaner = FolderCleaner(folders_to_clean, extensions=extensions_to_clean)
    cleaner.clean_folders()
