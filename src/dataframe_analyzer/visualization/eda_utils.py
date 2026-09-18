import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def plot_sales_chart(df, kind, x, y=None, title="", xlabel="", ylabel="", **kwargs):
    sns.set_style("whitegrid")
    plt.figure(figsize=kwargs.get("figsize", (12, 6)))
    
    if kind == "hist":
        sns.histplot(data=df, x=x, bins=kwargs.get("bins", 50), kde=True, color="skyblue", edgecolor="black")
    elif kind == "box":
        sns.boxplot(data=df, x=x, y=y, palette=kwargs.get("palette", "Set2"), fliersize=kwargs.get("fliersize", 5))
        
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()

def clean_dataframe(df, output_path="clean_train.csv"):
    print_header("PRIMERAS 5 FILAS")
    print(df.head())
    print_header("ULTIMAS 5 FILAS")
    print(df.tail())
    print_header("NOMBRES DE COLUMNAS")
    print(df.columns.tolist())
    print_header("INFORMACION GENERAL")
    df.info()
    print_header("ESTADISTICAS NUMERICAS")
    print(df.describe())
    print_header("CONTEO DE NULOS")
    print(df.isnull().sum())
    print("=" * 60)
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print(f"Duplicados: {df.duplicated().sum()}")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['week_day'] = df['date'].dt.weekday
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    return df