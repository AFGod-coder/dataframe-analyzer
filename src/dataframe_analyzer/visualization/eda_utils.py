"""
Helper functions for exploratory data analysis (EDA) and quick charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_chart(df, kind, x, y=None, title="", xlabel="", ylabel="", **kwargs):
    """Draw a quick chart (histogram or box plot) from a DataFrame.

    Args:
        df (pd.DataFrame): Data to plot.
        kind (str): Chart type, either "hist" or "box".
        x (str): Column name for the x-axis.
        y (str, optional): Column name for the y-axis. Only used for
            box plots. Defaults to None.
        title (str, optional): Chart title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "".
        **kwargs: Extra chart options: "figsize", "bins" (for
            histograms), "palette" and "fliersize" (for box plots).
    """
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