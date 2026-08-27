import os
import pandas as pd
from utils import clean_dataframe, plot_sales_histogram

ARCHIVO_ORIGINAL = "train.csv"

if __name__ == "__main__":
    if not os.path.exists(ARCHIVO_ORIGINAL):
        print(f"Error: Archivo '{ARCHIVO_ORIGINAL}' no encontrado.")
        exit(1)

    df_sucio = pd.read_csv(ARCHIVO_ORIGINAL)
    df_limpio = clean_dataframe(df_sucio)

    plot_sales_histogram(
        df=df_limpio,
        title="Distribucion de unidades vendidas",
        xlabel="Numero de unidades vendidas",
        ylabel="Frecuencia (dias)",
        bins=50
    )