import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, ElasticNet, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

from joblib import dump, load
import os


def regression_metrics(
        y_pred,
        y_test
) -> dict:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"MSE": mse, "MAE": mae, "R2": r2}


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
        dataset: pd.DataFrame = None,
        target_col: str = None,
        models: list = ["LinearRegression", "ElasticNet", "SVR", "BayesianRidge"],
        random_state: int = 42,
        test_size: float = 0.2,
        split_data: bool = True,
        scale_data: bool = True,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_train: pd.Series = None,
        y_test: pd.Series = None,
        save_models: bool = False,
        save_path: str = None
) -> dict:
    """Wrapper function that trains and evaluates multiple baseline regression
    models.

    Parameters
    ----------
    dataset: pd.DataFrame, default=None
        Dataset matrix.
    target_col: str, default=None
        The column containing the target values.
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
        # Check the types of the incoming data
        assert isinstance(dataset, pd.DataFrame)
        # Extract X and y
        X = dataset.drop(columns=target_col)
        y = dataset[target_col]
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
        metrics[model_name] = regression_metrics(y_pred, y_test)

        if save_models:
            if not save_path:
                raise ValueError(
                    "Please provide a path to save the models."
                )
            dump(model, os.path.join(
                save_path,
                f"{model_name}_baseline.joblib"
            ))
            if scale_data:
                dump(scaler, os.path.join(
                    save_path,
                    f"{model_name}_baseline_scaler.joblib"
                ))

    return metrics


class Regresssor:

    valid_types = {
        "LinearRegression",
        "ElasticNet",
        "SVR",
        "BayesianRidge"
    }

    valid_modes = {
        "baseline",
        "feature_selection",
        "tuning"
    }

    def __init__(
            self,
            model_type: str,
            dataset: pd.DataFrame,
            target_col: str,
            models_dir: str,
            random_state: int = 42
        ):

        if model_type not in self.valid_types:
            raise ValueError(
                f"Model must be one of the following:\n{self.valid_types}"
            )
        self.model_type = model_type

        assert isinstance(dataset, pd.DataFrame), "dataset must be <pd.DataFrame>"
        assert isinstance(target_col, str), "target_col must be <str>"
        self.X = dataset.drop(columns=target_col)
        self.y = dataset[target_col]

        assert isinstance(models_dir, str), "models_dir must be <str>"
        self.models_dir = models_dir

        assert isinstance(random_state, int), "random_state must be <int>"
        self.random_state = random_state


    def train(
            self,
            mode: str,
            test_size: float = 0.2,
            scale_data: bool = True,
            save_model:bool = False
        ):
        if mode not in self.valid_modes:
            raise ValueError(
                f"Mode must be one of the following:\n{self.valid_modes}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )

        if scale_data:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        # Simple baseline model
        if mode == "baseline":
                model_dict = {
                    "LinearRegression": LinearRegression(),
                    "ElasticNet": ElasticNet(random_state=self.random_state),
                    "SVR": SVR(),
                    "BayesianRidge": BayesianRidge()
                }
                model = model_dict[self.model_type]

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                metrics = self._regression_metrics(y_pred, y_test)

                if save_model:
                    dump(model, os.path.join(
                        self.models_dir,
                        f"{self.model_type}_{mode}.joblib"
                    ))
                    if scale_data:
                        dump(scaler, os.path.join(
                            f"{self.model_type}_{mode}_scaler.joblib"
                        ))

                return metrics

    def evaluate(
            self,
            mode: str,
            evaluation_set: pd.DataFrame,
            target_col: str
    ):
        pass

    def _regression_metrics(
            self,
            y_pred: pd.Series,
            y_test: pd.Series
    ) -> dict:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"MSE": mse, "MAE": mae, "R2": r2}