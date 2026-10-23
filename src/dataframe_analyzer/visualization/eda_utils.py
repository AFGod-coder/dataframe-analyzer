"""Charts for the exploratory data analysis (EDA).

This module has all the charts of the EDA stage of the project: sales
distribution by store and by product, and the sales trend over time.
"""

import logging
import os
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class EdaPlotter:
    """Create the charts of the exploratory data analysis.

    Each method builds one chart and can save it as a PNG file, so the
    chart can be used in a report without running the code again.

    Attributes:
        style (str): The Seaborn style applied globally to all plots.
    """

    def __init__(self, style: str = "whitegrid") -> None:
        """Set the Seaborn style for all charts.

        Args:
            style (str): Seaborn style name (e.g., 'whitegrid', 'darkgrid').
                Defaults to "whitegrid".
        """
        self.style = style
        sns.set_style(self.style)

    def _setup_figure(self, figsize: Tuple[int, int], title: str, xlabel: str, ylabel: str) -> None:
        """Set the size, title and axis labels of a new figure.

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

    def _finish(self, save_path: Optional[str] = None) -> None:
        """Save and/or show the current figure.

        Args:
            save_path (str, optional): If given, saves the figure to this
                path (creating parent folders as needed) before showing it.
                Defaults to None.
        """
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"Chart saved to: {save_path}")
        plt.show()
        plt.close()

    def plot_histogram(
        self,
        df: pd.DataFrame,
        x: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[int, int] = (12, 6),
        bins: int = 50,
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Draw a histogram with a KDE (density) curve.

        Args:
            df (pd.DataFrame): DataFrame containing the source data.
            x (str): Name of the column to plot on the x-axis.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            bins (int, optional): Number of histogram bins. Defaults to 50.
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (the chart is only shown).
            **kwargs (Any): Other keyword arguments, passed directly to
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
        self._finish(save_path)

    def plot_box(
        self,
        df: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[int, int] = (12, 6),
        palette: str = "Set2",
        fliersize: int = 5,
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Draw a box plot to study the distribution and the outliers.

        Args:
            df (pd.DataFrame): DataFrame containing the source data.
            x (str): Name of the column for the x-axis.
            y (str, optional): Name of the column for the y-axis.
                Defaults to None.
            hue (str, optional): Name of a category column that splits
                each box into groups with different colors. Use None for
                one simple box for each `x` category. `palette` colors the
                `hue` groups, not `x`. Defaults to None.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            palette (str, optional): Name of the Seaborn color palette. It
                is used only when `hue` is set. Defaults to "Set2".
            fliersize (int, optional): Size of the outlier markers. Defaults to 5.
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (the chart is only shown).
            **kwargs (Any): Other keyword arguments, passed directly to
                `sns.boxplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        sns.boxplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=palette if hue else None,
            fliersize=fliersize,
            **kwargs
        )
        self._finish(save_path)

    def plot_bar(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[int, int] = (12, 6),
        orient: str = "v",
        color: str = "skyblue",
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Draw a bar chart to compare categories.

        `df` must already be aggregated, with one row for each category
        (for example the result of `groupby(...).sum().reset_index()`).
        This method does not aggregate the data. It only draws it.

        Args:
            df (pd.DataFrame): Aggregated DataFrame (one row per category).
            x (str): Name of the column for the x-axis.
            y (str): Name of the column for the y-axis.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            orient (str, optional): "v" for vertical bars (few categories,
                for example stores) or "h" for horizontal bars (many
                categories, for example products), so the labels are easy
                to read. With "h", the method changes `x` and `y` itself:
                always give the category in `x` and the value in `y`.
                Defaults to "v".
            color (str, optional): Single color for all bars. Defaults to
                "skyblue".
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (the chart is only shown).
            **kwargs (Any): Other keyword arguments, passed directly to
                `sns.barplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        # Seaborn sorts numeric categories again if no `order` is given.
        # This would remove the order that `df` already has (for example
        # sorted by total sales). We pass the current row order as `order`
        # to keep it.
        category_order = df[x].tolist()
        if orient == "h":
            sns.barplot(data=df, x=y, y=x, color=color, orient="h", order=category_order, **kwargs)
            plt.xlabel(ylabel, fontsize=12)
            plt.ylabel(xlabel, fontsize=12)
        else:
            sns.barplot(data=df, x=x, y=y, color=color, orient="v", order=category_order, **kwargs)
        self._finish(save_path)

    def plot_line(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        figsize: Tuple[int, int] = (14, 6),
        marker: str = "o",
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Draw a line chart to study the trend and the seasonality.

        `df` must already be aggregated and in time order (for example
        sales added by month).

        Args:
            df (pd.DataFrame): Aggregated DataFrame ordered by the x-axis.
            x (str): Name of the column for the x-axis (usually a time unit).
            y (str): Name of the column for the y-axis (the aggregated metric).
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (14, 6).
            marker (str, optional): Marker style for each data point.
                Defaults to "o".
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (the chart is only shown).
            **kwargs (Any): Other keyword arguments, passed directly to
                `sns.lineplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        sns.lineplot(data=df, x=x, y=y, marker=marker, **kwargs)
        plt.xticks(rotation=45)
        self._finish(save_path)
