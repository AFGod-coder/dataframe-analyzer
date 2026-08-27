import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_header(title):
    print('\n' + '=' * 60)
    print(title)
    print('=' * 60)

def plot_sales_histogram(df, title="Histograma de Ventas", variable="sales", xlabel="Ventas",
                         ylabel="Frecuencia", bins=50):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=variable, bins=bins, kde=True,
                 color='skyblue', edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()

def plot_sales_histogram(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='store', y='sales', palette='Set2')

    plt.title('📦 Ventas por Tienda (Boxplot)', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Tienda', fontsize=12)
    plt.ylabel('Ventas (unidades)', fontsize=12)
    plt.show() 

def plot_sales_histogram(df, bestseller=True):
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df, x='item', y='sales', palette='viridis', fliersize=1)

    plt.title('📦 Ventas por Producto (Boxplot)', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Producto (Item)', fontsize=12)
    plt.ylabel('Ventas (unidades)', fontsize=12)
    plt.show()

def clean_dataframe(df):
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

    print('=' * 60)
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print(f"Duplicados: {df.duplicated().sum()}")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['week_day'] = df['date'].dt.weekday

    df.to_csv('clean_train.csv', index=False, encoding='utf-8')
    return df