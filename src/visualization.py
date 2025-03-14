import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, kendalltau


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


def plot_feature_spans(
        data: pd.DataFrame | np.ndarray,
        sort_features: bool = True,
        title: str = "Features"
    ) -> None:
    """Plot the mean, min, max, and IQR of each feature in the dataframe.

    Parameters
    ----------
    data : pandas DataFrame or numpy array
        The data to plot the feature spans for.

    sort_features : bool, default=True
        Whether to sort the features by their mean values.

    title : str, default="Features"
        The title of the plot.

    Returns
    -------
    None
    """

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise ValueError("Input must be a pandas DataFrame or a numpy array.")

    stats = np.array([
        np.mean(data, axis=0),
        np.min(data, axis=0),
        np.max(data, axis=0),
        np.percentile(data, 25, axis=0),
        np.percentile(data, 75, axis=0)
    ]).T

    if sort_features:
        stats = stats[stats[:, 0].argsort()]

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
            hoverinfo="none",
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
        xaxis_title="Feature Index",
        yaxis_title="Values",
        template="plotly_white",
        hovermode="x unified",
        height=600,
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
    methods. The correlation coefficients are then sorted based on the sum of the
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

    g = sns.pairplot(
        data,
        diag_kind="kde",
        plot_kws={"color": scatter_color},
        diag_kws={"color": kde_color}
    )

    if overlay_correlations:
        g.map_upper(annotate_correlations)
    g.map_lower(sns.kdeplot, levels=4, color="black")
    g.map_diag(plot_mean)

    plt.suptitle(title, y=1.02)
    plt.show()