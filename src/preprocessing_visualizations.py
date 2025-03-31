import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import display, clear_output
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import (
    mutual_info_regression,
    VarianceThreshold,
    mutual_info_regression
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def plot_kde(
        df: pd.Series | pd.DataFrame,
        title: str = "Kernel Density Estimation",
        x_label: str = "Values",
        y_label: str = "Frequency",
        hue_by: pd.Series | pd.DataFrame = None
    ) -> None:
    """Plot Kernel Density Estimation of the given data.

    Parameters
    ----------
    df : pd.Series or pd.DataFrame
        Data to plot.

    title : str, default="Kernel Density Estimation"
        Title of the plot.

    x_label : str, default="Values"
        Label for x-axis.

    y_label : str, default="Frequency"
        Label for y-axis.

    hue_by : pd.Series or pd.DataFrame, default=None
        Data to color the plot by.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 7))

    if isinstance(hue_by, pd.Series) or isinstance(hue_by, pd.DataFrame):
        data = pd.DataFrame({"BMI": df, "Color by": hue_by})
        ax = sns.histplot(
            data=data,
            x="BMI",
            bins=20,
            kde=True,
            hue="Color by",
            alpha=0.6
        )
        ax.get_legend().set_title("")
    else:
        sns.histplot(df, bins=20, kde=True)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


def plot_animated_scatter_notebook_only(
        Y: pd.Series,
        X: pd.DataFrame,
        t: float = 0.5,
        title: str = "Animated Scatter Plot of targets vs features",
    ) -> None:
    """Dynamically plot scatter plots of Y vs each column in X, updating every t
    seconds.
    
    Parameters
    ----------
    Y : pd.Series
        The target data to plot against the features.

    X : pd.DataFrame
        The features to plot against the target data.

    t : float, default=0.5
        The time interval between each plot.

    title : str, default="Animated Scatter Plot of targets vs features"
        The title of the plot.

    Returns
    -------
    None
    """
    for column in X.columns:
        clear_output(wait=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[column], Y, s=25, alpha=0.75, edgecolors="black")
        ax.set_xlabel(column)
        ax.set_ylabel(Y.name if Y.name else "Target")
        ax.set_title(title)
        
        display(fig)
        plt.close(fig)
        time.sleep(t)


def plot_specific_feature(
        df: pd.DataFrame,
        feature: str,
        target: pd.Series,
        title: str = "Scatter plot of {} vs {}",
        x_label: str = "Feature",
        y_label: str = "Target"
    ) -> None:
    """Plot the given feature against the target data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the feature and target data.

    feature : str
        The name of the feature to plot.

    target : pd.Series
        The target data to plot against the feature.

    title : str, default="Scatter plot of {} vs {}"
        The title of the plot.

    x_label : str, default="Feature"
        The label for the x-axis.

    y_label : str, default="Target"
        The label for the y-axis.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=feature, y=target)
    plt.title(title.format(feature, target.name))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_zeros(
        df: pd.DataFrame,
        target: pd.Series,
        title: str = 
            "Number of zeros in each feature and mean bmi for non-zero bacteria",
        hist_title: str = "Number of zeros in each feature",
        scatter_title: str = "Mean BMI for non-zero bacteria",
        y1_hline_title: str = "Total Samples ({})",
        y2_hline_title: str = "Mean BMI for all non-zero bacteria ({})",
        yaxis_1_title: str = "Number of zeros",
        yaxis_2_title: str = "Mean BMI"
    ) -> None:
    """Plot the number of zeros in each feature of the dataframe and the mean BMI
    of the non zero features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot the number of zeros for.

    bmi : pd.Series
        The BMI values of the samples.

    title : str, default="Number of zeros in each feature"
        The title of the plot.

    hist_title : str, default="Number of zeros in each feature"
        The title of the histogram.

    scatter_title : str, default="Mean BMI for non-zero bacteria"
        The title of the scatter plot.

    y1_hline_title : str, default="Total Samples ({})"
        The title of the first horizontal line.

    y2_hline_title : str, default="Mean BMI for all non-zero bacteria ({})"
        The title of the second horizontal line.

    yaxis_1_title : str, default="Number of zeros"
        The title of the y-axis for the histogram.

    yaxis_2_title : str, default="Mean BMI"
        The title of the y-axis for the scatter plot.

    Returns
    -------
    None
    """
    zeros = df.isin([0]).sum().sort_values(ascending=True)

    means = []
    for col in zeros.index:
        non_zero_indices = df[col] != 0
        mean_target = target[non_zero_indices].mean()
        means.append(mean_target)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=zeros.index,
        y=zeros.values,
        yaxis="y",
        name=hist_title
    ))

    fig.add_trace(go.Scatter(
        x=zeros.index,
        y=means,
        yaxis="y2",
        mode="markers",
        marker=dict(color="black", size=12),
        name=scatter_title
    ))

    fig.add_hline(
        y=df.shape[0],
        line_dash="dot",
        line_color="red",
        annotation_text=y1_hline_title.format(df.shape[0]),
        annotation_position="top left",
        annotation=dict(
            font_size=15,
            font_color="black",
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        )
    )

    fig.add_hline(
        y=target.mean(),
        yref="y2",
        line_dash="dot",
        line_color="red",
        annotation_text=y2_hline_title.format(target.mean()),
        annotation_position="top left",
        annotation=dict(
            font_size=15,
            font_color="black",
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        )
    )

    fig.update_layout(
        title_text=title,
        xaxis_title="Feature",
        xaxis_tickangle=45,
        yaxis_title=yaxis_1_title,
        yaxis2=dict(
            title=yaxis_2_title,
            overlaying="y",
            side="right"
        ),
        height=800
    )

    fig.show()


def plot_feature_spans(
        data: pd.DataFrame | np.ndarray,
        sort_features_by: bool | str = True,
        title: str = "Features"
    ) -> None:
    """Plot the mean, min, max, and IQR of each feature in the dataframe.

    Parameters
    ----------
    data : pandas DataFrame or numpy array
        The data to plot the feature spans for.

    sort_features_by : bool or str, default=False
        If True, the features are sorted by the mean value. If a string is
        passed, the features are sorted by the given statistic. The possible
        values are "mean", "min", "max", "q25", and "q75".

    title : str, default="Features"
        The title of the plot.

    Returns
    -------
    None
    """

    if isinstance(data, pd.DataFrame):
        feature_names = data.columns.tolist()
        data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
    else:
        raise ValueError("Input must be a pandas DataFrame or a numpy array.")

    stats = np.array([
        np.mean(data, axis=0),
        np.min(data, axis=0),
        np.max(data, axis=0),
        np.percentile(data, 25, axis=0),
        np.percentile(data, 75, axis=0)
    ]).T

    if isinstance(sort_features_by, str):
        sort_features_by = sort_features_by.lower()

    if sort_features_by:
        if sort_features_by == "mean" or sort_features_by == True:
            sorted_indices = stats[:, 0].argsort()
            stats = stats[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]
        elif sort_features_by == "min":
            sorted_indices = stats[:, 1].argsort()
            stats = stats[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]
        elif sort_features_by == "max":
            sorted_indices = stats[:, 2].argsort()
            stats = stats[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]
        elif sort_features_by == "q25":
            sorted_indices = stats[:, 3].argsort()
            stats = stats[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]
        elif sort_features_by == "q75":
            sorted_indices = stats[:, 4].argsort()
            stats = stats[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]
        else:
            raise ValueError(
                "".join([
                    "Invalid value for sort_features_by.",
                    "Must either 'mean', 'min', 'max', 'q25', 'q75', True or False."
                ])
            )
            
    feature_indices = list(range(len(stats)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=feature_indices, y=stats[:, 0], 
        mode="markers",
        marker=dict(color="black", size=12),
        zorder=3,
        name="Mean"
    ))

    for i in feature_indices:
        min_val = stats[i, 1]
        max_val = stats[i, 2]
        q25 = stats[i, 3]
        q75 = stats[i, 4]

        fig.add_trace(go.Scatter(
            x=[i, i],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#6f42f5", width=5, dash="solid"),
            showlegend=False,
            hoverinfo="none",
            name="Min to Max"
        ))

        fig.add_trace(go.Scatter(
            x=[i, i],
            y=[q25, q75],
            mode="lines",
            line=dict(color="#f54278", width=10, dash="solid", shape="spline"),
            opacity=0.5,
            showlegend=i == 0,
            zorder=2,
            name="IQR"
        ))

    fig.add_trace(go.Scatter(
        x=feature_indices, y=stats[:, 1], 
        mode="markers",
        marker=dict(color="#6f42f5", size=10),
        name="Min"
    ))

    fig.add_trace(go.Scatter(
        x=feature_indices, y=stats[:, 2], 
        mode="markers",
        marker=dict(color="#6f42f5", size=10),
        name="Max"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Values",
        template="plotly_white",
        hovermode="x unified",
        height=1000,
        xaxis=dict(
            tickmode='array',
            tickvals=feature_indices,
            ticktext=feature_names,
            tickangle=45
        )
    )

    fig.show()


def plot_correlation_coefficients(
        df: pd.DataFrame,
        against: pd.Series,
        abs_corr: bool = False,
        show_plot: bool = True,
        title: str = "Correlation Coefficients"
    ) -> tuple[np.ndarray, list]:
    """Calculate the correlation coefficients of the given dataframe against the
    given target data.

    The correlation coefficients are calculated using Pearson, Spearman, and Kendall
    tau methods. The correlation coefficients are then sorted based on the sum of the
    coefficients for each feature. The sorted features are plotted using a scatter
    plot.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the features to calculate the correlation
        coefficients. The columns of the dataframe are the features.

    against: pd.Series
        The target data to calculate the correlation coefficients against.

    abs_corr: bool, default=False
        If True, the absolute value of the correlation coefficients are taken.
        Otherwise, the raw correlation coefficients are used.

    show_plot: bool, default=True
        If True, the scatter plot of the sorted correlation coefficients is shown.

    title: str, default="Correlation Coefficients"
        The title of the plot.

    Returns
    -------
    corr_coeffs: np.ndarray
        The sorted correlation coefficients of the features against the target data.

    sorted_cols: list
        The sorted columns of the dataframe based on the correlation coefficients.
    """

    if isinstance(df, pd.Series):
        df = df.to_frame()

    corr_coeffs = np.zeros((df.shape[-1], 3))

    for i, col in enumerate(df.columns):
        pearsons_corr = pearsonr(against, df[col])
        spearmans_corr = spearmanr(against, df[col])
        kendalls_corr = kendalltau(against, df[col])

        corr_coeffs[i, 0] = abs(pearsons_corr[0]) if abs_corr else pearsons_corr[0]
        corr_coeffs[i, 1] = abs(spearmans_corr[0]) if abs_corr else spearmans_corr[0]
        corr_coeffs[i, 2] = abs(kendalls_corr[0]) if abs_corr else kendalls_corr[0]

    row_sums = np.sum(corr_coeffs, axis=1)
    sorted_row_indices = np.argsort(row_sums)[::-1]
    corr_coeffs = corr_coeffs[sorted_row_indices]

    x = list(df.columns)
    sorted_cols = [x[i] for i in sorted_row_indices]

    if not show_plot:
        return corr_coeffs, sorted_cols
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_cols,
        y=corr_coeffs[:, 0],
        mode="markers",
        marker=dict(size=12),
        name="Pearson"
    ))

    fig.add_trace(go.Scatter(
        x=sorted_cols,
        y=corr_coeffs[:, 1],
        mode="markers",
        marker=dict(size=12),
        name="Spearman"
    ))

    fig.add_trace(go.Scatter(
        x=sorted_cols,
        y=corr_coeffs[:, 2],
        mode="markers",
        marker=dict(size=12),
        name="Kendall"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Correlation Coefficient",
        template="plotly_white",
        height=700
    )

    fig.show()

    return corr_coeffs, sorted_cols


def plot_correlated_pairplot(
        data: pd.DataFrame,
        title: str = "Pairplot with Correlation Coefficients and KDE",
        scatter_color: str = "#7a4db0",
        kde_color: str = "#421f6e",
        hue: pd.Series = None,
        cmap: str = "viridis",
        overlay_correlations: bool = True
    ) -> None:
    """Creates a pairplot with Pearson, Spearman, and Kendall correlation
    coefficients overlaid on the upper triangle of the plot and the lower
    triangle replaced with kernel density estimates.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.

    title : str, default="Pairplot with Correlation Coefficients and KDE"
        The title of the plot.

    scatter_color : str, default="#55278c"
        The color of the scatter plot points.

    kde_color : str, default="#7d47bf"
        The color of the kernel density estimates.

    hue : pd.Series, default=None
        The target data to color the pairplot by.

    cmap : str, default="viridis"
        The colormap to use for coloring the pairplot.

    overlay_correlations : bool, default=True
        If True, the correlation coefficients are overlaid on the upper triangle of
        the pairplot.

    Returns
    -------
    None
    """

    def annotate_correlations(x, y, **kwargs):
        """Annotate the correlation coefficients on the pairplot."""

        pearson_coef, _ = pearsonr(x, y)
        spearman_coef, _ = spearmanr(x, y)
        kendall_coef, _ = kendalltau(x, y)

        text = "".join([
            f"Pearson: {pearson_coef:.2f}\n",
            f"Spearman: {spearman_coef:.2f}\n",
            f"Kendall: {kendall_coef:.2f}"
        ])
        
        plt.annotate(
            text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.75)
        )

    def plot_mean(x, **kwargs):
        """Plot the mean of the data on the diagonal of the pairplot."""

        mean_val = np.mean(x)
        plt.axvline(
            mean_val,
            color="black",
            linestyle="--",
            label=f"Mean: {mean_val:.2f}"
        )
        plt.legend()

    if hue is not None:
        plot_kws = {"hue": hue, "palette": cmap}
    else:
        plot_kws = {"color": scatter_color}

    g = sns.pairplot(
        data,
        diag_kind="kde",
        plot_kws=plot_kws,
        diag_kws={"color": kde_color}
    )

    if overlay_correlations:
        g.map_upper(annotate_correlations)
    g.map_lower(sns.kdeplot, levels=4, color="black")
    g.map_diag(plot_mean)


    # Add a color bar if hue is provided
    if hue is not None:
        # Create a normalized scalar mappable for the color bar
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        g.figure.subplots_adjust(right=0.85)
        cbar_ax = g.figure.add_axes([0.88, 0.15, 0.02, 0.7])
        g.figure.colorbar(sm, cax=cbar_ax, label=hue.name)

    plt.suptitle(title, y=1.02)
    plt.show()


def calc_feature_selection_metrics(
        X: pd.DataFrame,
        y: pd.Series,
        method: str  
    ) -> pd.DataFrame:
    """Calculate the feature selection method performance metrics.

    Parameters
    ----------
    X : pd.DataFrame
        The feature data.

    y : pd.Series
        The target data.

    methods : str
        The feature selection method used.

    Returns
    -------
    feature_selection_metrics : pd.DataFrame
        DataFrame containing the feature selection method performance metrics.
    """

    def compute_vif(X):
        if X.shape[1] > 1:
            return pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            ).mean()
        else:
            return 0 

    selector = VarianceThreshold(threshold=0.1)
    selector.fit(X)

    new_row = pd.DataFrame({
        "Method": [
            method
        ],
        "Number of Features": [
            X.shape[1]
        ],
        "Mutual Information": [
            mutual_info_regression(X, y).mean()
        ],
        "Variance": [
            X[X.columns[selector.get_support()]].var().mean()
        ],
        "Variance Inflation Factor": [
            compute_vif(X)
        ]
    })

    return new_row


def plot_feature_selection_metrics(
        feature_selection_metrics: pd.DataFrame,
        shape: tuple,
        figsize: tuple = (15, 10)
    ) -> None:
    """
    Plot the feature selection method performance by metric.

    Parameters
    ----------
    feature_selection_metrics : pd.DataFrame
        DataFrame containing the feature selection method performance metrics.

    shape: tuple
        Shape of the plot grid.

    figsize: tuple default = (15, 10)
        Size of the figure.        

    Returns
    -------
    None
    """

    metrics = feature_selection_metrics.columns[1:]

    k, l = shape

    fig, axes = plt.subplots(k, l, figsize=figsize)

    axes = axes.flatten()

    fig.suptitle(
        "Feature Selection Method Performance by Metric",
        fontsize=14,
        y=1.05
    )

    palette = sns.color_palette(
        "husl",
        n_colors=len(feature_selection_metrics["Method"])
    )

    for i, metric in enumerate(metrics):
        sns.barplot(
            data=feature_selection_metrics,
            x="Method",
            y=metric,
            ax=axes[i],
            palette=palette,
            edgecolor="black",
            linewidth=0.5
        )
        
        axes[i].set_title(metric, fontsize=12)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Score", fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        
        for p in axes[i].patches:
            height = p.get_height()
            axes[i].annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=9
            )

    plt.tight_layout()
    plt.show()