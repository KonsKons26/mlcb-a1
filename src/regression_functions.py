import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def baseline_linear_regression(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        split_data: bool = True,
        scale_data: bool = True,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_train: pd.Series = None,
        y_test: pd.Series = None
    ) -> tuple[LinearRegression, dict, pd.DataFrame]:
    """Wrapper for the least squares Linear Regression sklearn class, used as a
    baseline model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.
    split_data : bool, default=True
        Whether to split the data into training and test sets.
    scale_data : bool, default=True
        Whether to scale the data.
    X_train : pd.DataFrame, default=None
        Training feature matrix.
    X_test : pd.DataFrame, default=None
        Test feature matrix.
    y_train : pd.Series, default=None
        Training target variable.
    y_test : pd.Series, default=None
        Test target variable.

    Returns
    -------
    model : LinearRegression
        Fitted linear regression model.
    metrics : dict
        Dictionary containing the mean squared error (MSE) and R-squared (R2) score.
    coefficients : pd.DataFrame
        DataFrame containing the coefficients of the linear regression model.
    residuals : pd.Series
        Residuals of the model predictions.
    """

    # Check the types of the incoming data
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # Check that the data are presplit or if the function will split them
    if (
        split_data and any([X_train, X_test, y_train, y_test])
    ) or (
        not split_data and not all([X_train, X_test, y_train, y_test])
    ):
        raise ValueError(
            "Either pass all the pre-split dataset: "
            "'X_train, X_test, y_train, y_test'\n"
            "Or let the function perform the split 'split_data=True'."
        )

    # Split the data
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Scale the data
    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

    # Model training
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "MSE": mse,
        "R2": r2
    }

    # Feature contributions
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)

    return model, metrics, coefficients