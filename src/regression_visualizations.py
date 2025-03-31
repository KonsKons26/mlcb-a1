import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go



def plot_metrics(
        metrics: dict,
        figsize: tuple = (18, 6),
        models: list = ["ElasticNet", "SVR", "BayesianRidge"],
        metrics_list: list = ["MAE", "MSE", "R2"]
    ) -> None:
    """

    """

    metrics_by_metric = {metric: {model: [] for model in models} for metric in metrics_list}

    # metrics_by_metric = {}

    for model, metrics_dict in metrics.items():
        if model in models:
            for metric, values_list in metrics_dict.items():
                if metric in metrics_list:
                    metrics_by_metric[metric][model] = values_list
    
    fig, axes = plt.subplots( nrows=1, ncols=3, figsize=figsize)

    for i, ax in enumerate(axes):
        metric = list(metrics_by_metric.keys())[i]
        values = metrics_by_metric[metric]
        sns.boxplot(data=pd.DataFrame(values), ax=ax)
        ax.set_title(f"{metric} by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.grid(True)
    plt.tight_layout()
    plt.show()