"""Module for exploratory data analysis (EDA) and professional visualizations.

This module centralizes every chart used during the EDA stage of the
project (Hito 2, section 2.1 - Dataset Construction): distribution of
sales by store, by product, and the sales trend over time.
"""

import logging
import os
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class EdaPlotter:
    """A professional plotter class for Exploratory Data Analysis.

    Every method builds one chart and, optionally, saves it as a PNG
    file so it can be reused in the article/report without having to
    re-run the code.

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

    def _finish(self, save_path: Optional[str] = None) -> None:
        """Internal helper to save and/or show the current figure.

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
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (chart is only shown, not saved).
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
        """Draws a box plot to analyze distributions and outliers.

        Args:
            df (pd.DataFrame): DataFrame containing the source data.
            x (str): Name of the column for the x-axis.
            y (str, optional): Name of the column for the y-axis.
                Defaults to None.
            hue (str, optional): Name of a categorical column used to
                split/color each box further. Leave as None for a plain
                box plot per `x` category (this is what makes `palette`
                meaningful: it colors the `hue` groups, not `x` itself).
                Defaults to None.
            title (str, optional): Title text for the chart. Defaults to "".
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            figsize (Tuple[int, int], optional): Dimensions of the figure.
                Defaults to (12, 6).
            palette (str, optional): Color palette name for Seaborn. Only
                takes effect when `hue` is set. Defaults to "Set2".
            fliersize (int, optional): Size of the outlier markers. Defaults to 5.
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (chart is only shown, not saved).
            **kwargs (Any): Additional keyword arguments passed directly to
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
        """Draws a bar plot for categorical comparisons.

        `df` is expected to already be aggregated (one row per category,
        e.g. the result of `groupby(...).sum().reset_index()`): this
        method does not aggregate data itself, it only draws it.

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
                e.g. stores) or "h" for horizontal bars (many categories,
                e.g. products) so labels stay readable. When "h", `x` and
                `y` are swapped internally: pass them the same way as for
                "v" (category in `x`, value in `y`) and this method takes
                care of it. Defaults to "v".
            color (str, optional): Single color for all bars. Defaults to
                "skyblue".
            save_path (str, optional): Path to save the chart as a PNG.
                Defaults to None (chart is only shown, not saved).
            **kwargs (Any): Additional keyword arguments passed directly to
                `sns.barplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        # Seaborn re-sorts numeric-looking categories on its own unless an
        # explicit `order` is given, which would silently undo any sorting
        # already applied to `df` (e.g. sort_values by total sales). Passing
        # the category column's current row order as `order` preserves it.
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
        """Draws a line chart, used for trend and seasonality analysis.

        `df` is expected to already be aggregated in time order (e.g. the
        result of grouping by month and summing sales).

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
                Defaults to None (chart is only shown, not saved).
            **kwargs (Any): Additional keyword arguments passed directly to
                `sns.lineplot`.
        """
        self._setup_figure(figsize, title, xlabel, ylabel)
        sns.lineplot(data=df, x=x, y=y, marker=marker, **kwargs)
        plt.xticks(rotation=45)
        self._finish(save_path)
