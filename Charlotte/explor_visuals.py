import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

sns.set_style("whitegrid")


def Countplot_Visual(
    data: Union[pd.Series, np.array, pd.DataFrame, np.ndarray],
    hue: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    title: str | None = None,
) -> plt.Figure:
    """
    Visualize counts of categorical data using Seaborn's countplot.

    Args:
        data (Union[pd.Series, np.array, pd.DataFrame, np.ndarray]): Input data for visualization.
        hue (str | None, optional): Categorical variable for color encoding. Defaults to None.
        figsize (tuple[int,int], optional): Figure size (width, height) in inches. Defaults to (12,6).
        title (str | None, optional): Title for the plot. Defaults to None.

    Returns:
        plt.figure: Matplotlib Figure object displaying the countplot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, pd.Series) or isinstance(data, np.ndarray):
        sns.countplot(x=data, hue=hue, ax=ax)
        ax.set_title(
            data.name
            if hasattr(data, "name")
            else "Count Of ..." if title is None else title
        )
    elif isinstance(data, pd.DataFrame):
        if hue is None:
            sns.countplot(data=data, x=data.columns[0], ax=ax)
            ax.set_title(f"Count of {data.columns[0]}" if title is None else title)
        else:
            sns.countplot(data=data, x=data.columns[0], hue=hue, ax=ax)
            ax.set_title(
                f"Count by {hue} grouped by {data.columns[0]}"
                if title is None
                else title
            )
        for p in ax.patches:
            ax.annotate(
                f"\n{p.get_height()}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="top",
                color="black",
            )

    return fig


def PiePlot(
    data: pd.Series,
    figsize: tuple[int, int] = (12, 6),
    title: str | None = None,
    palatte: str | None = None,
) -> plt.Figure:
    """_summary_

    Args:
        data (pd.Series): _description_
        figsize (tuple[int,int], optional): _description_. Defaults to (12,6).
        title (str | None, optional): _description_. Defaults to None.
        palatte (str | None, optional): _description_. Defaults to None.

    Returns:
        plt.Figure: _description_
    """
    fig, ax = plt.subplots(figsize=figsize)
    palette_color = sns.color_palette("bright") if palatte is None else palatte
    if isinstance(data, pd.Series):
        data_count = data.value_counts()
        keys = data_count.index  # Use the index of value_counts instead of unique()
        ax.pie(data_count, labels=keys, colors=palette_color, autopct="%.1f%%")

    if title:
        ax.set_title(title)

    return fig
