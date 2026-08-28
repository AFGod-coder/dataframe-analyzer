"""
DataLoader class.
Loads a CSV file from the given path.
"""

import os
import pandas as pd

class DataLoader:
    """
    Loads a CSV file and returns a pandas DataFrame.
    """

    def __init__(self, file_path: str):
        """
        Saves the file path.

        Parameters:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path

    def load(self):
        """
        Loads the CSV file.

        Returns:
            pd.DataFrame: Data from the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._validate_file_exists()
        return pd.read_csv(self.file_path)

    def _validate_file_exists(self):
        """
        Checks if the file exists on disk.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")