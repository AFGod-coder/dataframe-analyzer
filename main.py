import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from DataLoader import DataLoader
from DataCleaner import DataCleaner
from DataTransform import DataTransform, DatePart

FILE_PATH = "train.csv"

if __name__ == "__main__":
    
    loader = DataLoader(FILE_PATH)
    df_raw = loader.load()
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
    
    df_clean = DataTransform.transform_to_date(df_clean, 'date')
    df_clean = DataTransform.add_date_part_column(df_clean, 'date', DatePart.DAY)
    df_clean = DataTransform.add_date_part_column(df_clean, 'date', DatePart.MONTH)
    df_clean = DataTransform.add_date_part_column(df_clean, 'date', DatePart.YEAR)
    df_clean = df_clean.drop(columns=['date'])
    
    df_clean.to_csv('clean_dataframe.csv', index=False)