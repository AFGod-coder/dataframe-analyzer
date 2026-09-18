"""
Loads a CSV file into a pandas DataFrame.
"""

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads a CSV file from disk and returns it as a DataFrame."""

    def __init__(self, file_path: str):
        """Store the path to the CSV file.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Read the CSV file and return its content.

        Returns:
            pd.DataFrame: The data loaded from the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        logger.info(f"Loading file: {self.file_path}")
        self._validate_file_exists()
        df = pd.read_csv(self.file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        return df

    def _validate_file_exists(self):
        """Check that the file exists on disk.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
