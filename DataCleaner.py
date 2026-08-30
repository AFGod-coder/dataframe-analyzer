import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, date_col: str = None, text_cols: list = None):
        self.date_col = date_col
        self.text_cols = text_cols if text_cols is not None else []

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting cleaning pipeline...")
        df_clean = df.copy()
        
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        logger.info(f"Removed {removed} duplicate rows.")  
        
        
        df_clean = self._remove_string_spaces(df_clean)
        logger.info(f"Removed unnecesary spaces in column type string.")  
        logger.info("Cleaning finished successfully.")
        return df_clean
    
    def _remove_string_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_cols:
            for col in self.text_cols:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].str.strip()
        else:
            text_columns = df.select_dtypes(include=['object']).columns
            for col in text_columns:
                df[col] = df[col].str.strip()
        return df