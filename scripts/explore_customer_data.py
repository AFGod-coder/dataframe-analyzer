import pandas as pd

from dataframe_analyzer.data.transform import DataTransform

FILE_PATH = 'data/raw/customer_shopping_data.csv'

if __name__ == "__main__":
    df_raw = pd.read_csv(FILE_PATH)

    df_raw.info()
    print(df_raw.head())

    df_raw = DataTransform.transform_to_date(df_raw, 'invoice_date')
    df_raw.info()

    print(df_raw.describe())
