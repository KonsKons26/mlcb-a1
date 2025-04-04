import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def pretty_print_metrics(metrics: dict) -> str:
    """Function to tabulate (in a way) the results of the pipeline."""
    string = ""
    string += "========================================================\n"
    for model in ["ElasticNet", "SVR", "BayesianRidge"]:
        string += "\t\t--------------------\n"
        string += f"\t\t{model:^20}\n"
        string += "\t\t--------------------\n"

        for mode in ["baseline", "feature_selection", "tune"]:
            string += f"{mode}:\n"
            string += "\ttraining\t\t\tvalidation\n"
            string += "\t--------\t\t\t----------\n"

            for (ktrain, vtrain), (kval, vval) in zip(
                metrics["training"][model][mode].items(),
                metrics["validation"][model][mode].items()
            ):
                string += f"\t{ktrain}:\t{np.mean(vtrain):.4f}\t\t\t{kval}:\t{np.mean(vval):.4f}\n"

            if mode == "feature_selection":
                string += f"\n\tFeature selection method:\n\t\t\t{metrics["training"][model]["feature_selection"]["feature_selection_method"]}\n"
                string += f"\tNumber of features used:\n\t\t\t{len(metrics["training"][model]["feature_selection"]["features"])}\n"

            if mode == "tune":
                string += "\n\tBest hyperparameters:\n"
                for k, v in metrics["training"][model]["tune"]["best_hyperparameters"].items():
                    string += f"\t\t\t{k}: {v}\n"

            string += "\n"
        string += "=========================================================\n"
    return string


def plot_metrics(
        all_metrics,
        tr_or_val = "validation",
        models = ["ElasticNet", "SVR", "BayesianRidge"],
        modes = ["baseline", "feature_selection", "tune"],
        metrics = ["RMSE", "MAE", "R2"]
    ) -> None:
    """Plot RMSE, MAE, and R2 for each model grouped by mode"""

    def plot_metrics_helper(
        model,
        metric,
        df
    ):
        plt.figure(figsize=(10, 10))
        sns.boxplot(
            data=df
        )
        plt.title(model)
        plt.ylabel(metric)
        plt.show()

    
    metrics_dict = {
        model: {
            metric: {
                mode: [] for mode in modes
            } for metric in metrics
        } for model in models
    }

    for model in models:
        for mode in modes:
            for metric in metrics:
                metrics_dict[model][metric][mode].extend(all_metrics[tr_or_val][model][mode][metric])


    for model in models:
        for metric in metrics:
            df = pd.DataFrame(columns=modes)
            for mode in modes:
                df[mode] = metrics_dict[model][metric][mode]
            plot_metrics_helper(
                model,
                metric,
                df
            )



def plot_features_interactive(all_features, models_dir):
    """Plot the features selected across all models with model-based color
    segments."""

    models = ["ElasticNet", "SVR", "BayesianRidge"]

    feat_dict = {f: {model: 0 for model in models} for f in all_features}

    for model in models:
        feature_file = os.path.join(models_dir, f"{model}_features.txt")
        with open(feature_file, "r") as handle:
            fs = [line.strip() for line in handle]
        for f in fs:
            if f in feat_dict:
                feat_dict[f][model] += 1

    feat_dict = {
        f: counts for f, counts in feat_dict.items()
        if sum(counts.values()) > 0
    }

    sorted_items = sorted(feat_dict.items(), key=lambda item: sum(item[1].values()), reverse=True)
    features = [item[0] for item in sorted_items]

    data = {model: [feat_dict[feat][model] for feat in features] for model in models}

    fig = go.Figure()

    colors = {
        "ElasticNet": "indigo",
        "SVR": "teal",
        "BayesianRidge": "tomato"
    }

    for model in models:
        fig.add_trace(go.Bar(
            x=features,
            y=data[model],
            name=model,
            marker_color=colors.get(model, None)
        ))

    fig.update_layout(
        barmode='stack',
        title=dict(
            text="Feature Frequency by Model",
            font=dict(size=24)
        ),
        xaxis_title=dict(
            text="Features",
            font=dict(size=20)
        ),
        yaxis_title=dict(
            text="Count",
            font=dict(size=20)
        ),
        template="plotly_white",
        height=1000,
        legend=dict(
            font=dict(size=18)
        )
    )

    fig.show()