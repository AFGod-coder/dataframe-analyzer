"""Module for exploratory data analysis (EDA) and professional visualizations."""

from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EdaPlotter:
    """A professional plotter class for Exploratory Data Analysis.

    Attributes:
        style (str): The Seaborn style applied globally to all plots.
    """

    def __init__(self, style: str = "whitegrid") -> None:
        """Initializes the plotter with a default Seaborn style.

        Args:
            style (str): Seaborn style name (e.g., 'whitegrid', 'darkgrid').
                Defaults to "whitegrid".
        """
        self.style = style
        sns.set_style(self.style)

    def _setup_figure(self, figsize: Tuple[int, int], title: str, xlabel: str, ylabel: str) -> None:
        """Internal helper to configure common plot elements (DRY principle).

        Args:
            figsize (Tuple[int, int]): Width and height of the figure in inches.
            title (str): Title text for the chart.
            xlabel (str): Label text for the x-axis.
            ylabel (str): Label text for the y-axis.
        """
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

    def plot_histogram(
        self, 
        df: pd.DataFrame, 
        x: str, 
        title: str = "", 
        xlabel: str = "", 
        ylabel: str = "", 
        figsize: Tuple[int, int] = (12, 6),
        bins: int = 50,
        **kwargs: Any
    ) -> None:
        """Draws a quick and clean histogram with a KDE curve.

        Args:
            df (pd.DataFrame): DataFrame containing the source data.
            x (str): Name of the column to plot on the x-axis.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            bins (int, optional): Number of histogram bins. Defaults to 50.
            **kwargs (Any): Additional keyword arguments passed directly to
                `sns.histplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        sns.histplot(
            data=df, 
            x=x, 
            bins=bins, 
            kde=True, 
            color="skyblue", 
            edgecolor="black", 
            **kwargs
        )
        plt.show()

    def plot_box(
        self, 
        df: pd.DataFrame, 
        x: str, 
        y: Optional[str] = None, 
        title: str = "", 
        xlabel: str = "", 
        ylabel: str = "", 
        figsize: Tuple[int, int] = (12, 6),
        palette: str = "Set2",
        fliersize: int = 5,
        **kwargs: Any
    ) -> None:
        """Draws a box plot to analyze distributions and outliers.

        Args:
            df (pd.DataFrame): DataFrame containing the source data.
            x (str): Name of the column for the x-axis.
            y (str, optional): Name of the column for the y-axis.
                Defaults to None.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            palette (str, optional): Color palette name for Seaborn.
                Defaults to "Set2".
            fliersize (int, optional): Size of the outlier markers. Defaults to 5.
            **kwargs (Any): Additional keyword arguments passed directly to
                `sns.boxplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        sns.boxplot(
            data=df, 
            x=x, 
            y=y, 
            palette=palette, 
            fliersize=fliersize, 
            **kwargs
        )
        plt.show()

    def plot_bar(
        self, 
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str = "", 
        xlabel: str = "", 
        ylabel: str = "", 
        figsize: Tuple[int, int] = (12, 6),
        **kwargs: Any
    ) -> None:
        """Draws a bar plot for categorical comparisons.

        Args:
            df (pd.DataFrame): DataFrame containing the source data.
            x (str): Name of the column for the x-axis.
            y (str): Name of the column for the y-axis.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            **kwargs (Any): Additional keyword arguments passed directly to
                `sns.barplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        sns.barplot(data=df, x=x, y=y, **kwargs)
        plt.show()
