import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, ElasticNet, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score



def baseline_train_test(
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series
) -> pd.Series:
    """Function to train and test a regression model.

    Parameters
    ----------
    model : object
        Regression model to be trained and tested.
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    y_train : pd.Series
        Training target variable.

    Returns
    -------
    y_pred : pd.Series
        Predicted target variable for the test set.
    """

    # Fit the model and test it with no parmeters
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred


def baseline_wrapper(
        X: pd.DataFrame,
        y: pd.Series,
        models: list = ["LinearRegression", "ElasticNet", "SVR", "BayesianRidge"],
        random_state: int = 42,
        test_size: float = 0.2,
        split_data: bool = True,
        scale_data: bool = True,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_train: pd.Series = None,
        y_test: pd.Series = None
) -> dict:
    """Wrapper function that trains and evaluates multiple baseline regression models.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    models : list, default=["LinearRegression", "ElasticNet", "SVR", "BayesianRidge"]
        List of models to train and evaluate.
    random_state : int, default=42
        Random state for reproducibility.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
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
    metrics : dict
        Dictionary containing the mean squared error (MSE) and R-squared (R2) score
        for each model.
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
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Initialize the models
    model_dict = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(random_state=random_state),
        "SVR": SVR(),
        "BayesianRidge": BayesianRidge()
    }
    models = {model: model_dict[model] for model in models if model in model_dict}
    metrics = {model: {} for model in models}

    # Train and evaluate each model
    for model_name, model in models.items():
        y_pred = baseline_train_test(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train
        )
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[model_name]["MSE"] = mse
        metrics[model_name]["R2"] = r2

    return metrics
