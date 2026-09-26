import pandas as pd
from dataframe_analyzer.visualization.eda_utils import EdaPlotter

FILE_PATH = "data/processed/clean_dataframe.csv"

if __name__ == "__main__":
    
    df_clean = pd.read_csv(FILE_PATH)
    
    df_grouped = df_clean.groupby(['store', "item"]).size()
    
    df_grouped = df_clean.groupby('store')['sales'].sum().reset_index()
    print(f"Sales by Store: \n{df_grouped}")

    print(f"Show grafic boxplot of Sales by store")
    plotter = EdaPlotter(style="whitegrid")
    
    plotter.plot_bar(
        df=df_clean, 
        x='store', 
        y='sales', 
        title='Distribution of Sales by Store', 
        xlabel='Stores',
        ylabel='Sales for each store',
        palette="Set2",
        hue='year' 
    ) 