import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def plot_metrics(
        metrics: dict[str, dict[str, list | float]],
        plots: str,
        figsize: tuple = (18, 6),
        models: list = ["ElasticNet", "SVR", "BayesianRidge"],
        metrics_list: list = ["MAE", "MSE", "R2"],
        linewidth: float = 2,
        title_fontsize: int = 16
    ) -> None:
    """
    Plot metrics with customizable line thickness and title sizes.
    
    Parameters
    ----------
    metrics: dict[str, dict[str, list | float]]
        The dictionary containing the metrics per model.
    plots: str
        Whether to use boxplots or barplots.
    figsize: tuple, default = (18, 6)
        The size of the plotted figure.
    models: list, default = ["ElasticNet", "SVR", "BayesianRidge"]
        The list of models to plot.
    metrics_list: list, default = ["MAE", "MSE", "R2"]
        The list of metrics to plot.
    linewidth: float, default = 2
        The linewidth of the edges of the boxes/ bars.
    title_fontsize: int, default = 16
        The fontsize of the title.

    Returns
    -------
    None

    Notes
    -----
    - Set boxplots to True if there are many values per metric, if each metric has
        a single value, set it to False.
    """

    # Check plot called
    if plots != "boxplot" and plots != "barplot":
        raise ValueError(
            "Either set 'plots' to 'boxplot' or 'barplot'."
        )

    # Convert the metrics dictionary from {model: {metric: value | values}} to
    # {metric: {model: value | values}} to plot based on metric
    metrics_by_metric = {
        metric: {model: [] for model in models} for metric in metrics_list
    }
    for model, metrics_dict in metrics.items():
        if model in models:
            for metric, values_list in metrics_dict.items():
                if metric in metrics_list:
                    metrics_by_metric[metric][model] = values_list

    # Plotting
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    for i, ax in enumerate(axes):
        metric = list(metrics_by_metric.keys())[i]
        values = metrics_by_metric[metric]

        if plots == "boxplot":
            # Boxplot with thicker lines
            boxplot = sns.boxplot(
                data=values,
                ax=ax,
                color="#2a59a3",
                linewidth=linewidth
            )
            # Make median line thicker
            for artist in boxplot.artists:
                artist.set_edgecolor('black')
                artist.set_linewidth(linewidth)
        elif plots == "barplot":
            # Barplot with thicker lines
            barplot = sns.barplot(
                data=values,
                ax=ax,
                color="#2a59a3",
                linewidth=linewidth
            )
            # Make outline thicker
            for patch in barplot.patches:
                patch.set_edgecolor('black')
                patch.set_linewidth(linewidth)

        ax.set_title(metric, fontsize=title_fontsize)

        # Make axis lines thicker
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth/2)

    plt.tight_layout()
    plt.show()