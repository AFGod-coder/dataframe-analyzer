import os
import pandas as pd
from utils import clean_dataframe, plot_sales_histogram

ARCHIVO_ORIGINAL = "train.csv"
ARCHIVO_LIMPIO = "clean_train.csv"

if __name__ == "__main__":
    if not os.path.exists(ARCHIVO_ORIGINAL):
        print(f"Error: Archivo '{ARCHIVO_ORIGINAL}' no encontrado.")
        exit(1)

    try:
        df_sucio = pd.read_csv(ARCHIVO_ORIGINAL)
        
        
        
    except FileNotFoundError:
        print("El archivo no se encuentra en el directorio")
        
    try:
        df_limpio = pd.read_csv(ARCHIVO_LIMPIO)
    except FileNotFoundError:
        df_limpio = clean_dataframe(df_sucio)