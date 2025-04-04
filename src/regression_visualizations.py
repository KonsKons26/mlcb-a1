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

    def plot_metrics_helper(
        model,
        metric,
        df
    ):
        plt.figure(figsize=(10, 10))
        sns.boxplot(
            data=df
        )
        plt.title(f"{model} - {metric}")
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