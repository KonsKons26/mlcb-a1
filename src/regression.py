import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    ElasticNet, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

from joblib import dump, load
import os


class Regresssor:
    """

    """

    valid_types = {
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
        ) -> dict:
        """Trains the model based on the specified mode and evaluates its
        performance.
            
        Parameters
        ----------
        mode : str
            The mode of training. Must be one of the valid modes defined in
            `self.valid_modes`.
        test_size : float, defaul = 0.2
            The proportion of the dataset to include in the test split.
        scale_data : bool, default = True
            Whether to scale the input data using StandardScaler.
        save_model : bool, default = False
            Whether to save the trained model and scaler (if used).

        Returns
        -------
        dict
            A dictionary containing regression metrics such as mean squared error,
            mean absolute error,  and R-squared score for the trained model on the
            test set.

        Raises
        ------
        ValueError
            If the specified mode is not in `self.valid_modes`.

        Notes
        -----
        - The method splits the data into training and testing sets using
            `train_test_split`.
        - If `scale_data` is True, the input features are scaled using
            `StandardScaler`.
        - Supports multiple baseline regression models such as LinearRegression,
            ElasticNet, SVR, and BayesianRidge.
        - Saves the trained model and scaler (if used) to the directory specified by
            `self.models_dir` if `save_model` is True.
        """

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
            X_train = pd.DataFrame(
                scaler.transform(X_train), columns=X_train.columns
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns
            )
            self.scaler = scaler

        if mode == "baseline":
                model_dict = {
                    "ElasticNet": ElasticNet(random_state=self.random_state),
                    "SVR": SVR(),
                    "BayesianRidge": BayesianRidge()
                }
                model = model_dict[self.model_type]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if save_model:
                    dump(model, os.path.join(
                        self.models_dir,
                        f"{self.model_type}_{mode}.joblib"
                    ))
                    if scale_data:
                        dump(scaler, os.path.join(
                            f"{self.model_type}_{mode}_scaler.joblib"
                        ))

                self.model = model

                return self._regression_metrics(y_pred, y_test)


    def evaluate(
            self,
            load_model: bool,
            evaluation_df: pd.DataFrame,
            target_col: str,
            model_name: str = None,
            scaled: bool = True,
            eval_loops: int = 50
        ) -> dict:
        """Evaluates the model using K-Fold cross-validation.
        
        Parameters
        ----------
        load_model : bool
            Whether to load a pre-trained model from disk.
        evaluation_df : pd.DataFrame
            The DataFrame containing the evaluation data.
        target_col : str
            The name of the target column in the evaluation DataFrame.
        model_name : str, optional
            The name of the model to load. Required if `load_model` is True.
        scaled : bool, default = True
            Whether the input data was scaled during training.

        Returns
        -------
        dict
            A dictionary containing regression metrics such as mean squared error,
            mean absolute error, and R-squared score for the model on the evaluation
            data.

        Raises
        ------
        ValueError
            If `load_model` is True and `model_name` is not provided.
        AssertionError
            If `evaluation_df` is not a DataFrame or if `target_col` is not a string.

        Notes
        -----
        - The method splits the evaluation data into K folds using `KFold`.
        - If `load_model` is True, it loads the pre-trained model and scaler (if
            used) from the specified directory.
        - If `scaled` is True, it scales the evaluation data using the loaded scaler.
        - Returns a dictionary containing the mean squared error, mean absolute
            error, and R-squared score for each fold.
        """
        X_val = evaluation_df.drop(columns=target_col)
        y_val = evaluation_df[target_col]

        if load_model:
            model = load(os.path.join(
                self.models_dir,
                model_name,
                ".joblib"
            ))
            if scaled:
                scaler = load(os.path.join(
                    self.models_dir,
                    model_name,
                    "_scaler",
                    ".joblib"
                ))
                X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        else:
            model = self.model
            if scaled and hasattr(self, "scaler"):
                X_val = pd.DataFrame(
                    self.scaler.transform(X_val), columns=X_val.columns
                )

        kf = KFold(n_splits=eval_loops)

        all_metrics = {"MSE": [], "MAE": [], "R2": []}

        for (X_idx, _), (y_idx, _) in zip(kf.split(X_val), kf.split(y_val)):
            X = X_val.iloc[X_idx]
            y = y_val.iloc[y_idx]
            y_pred = model.predict(X)
            metrics = self._regression_metrics(y_pred, y)
            for metric, value in metrics.items():
                all_metrics[metric].append(value)

        return all_metrics


    def _regression_metrics(
            self,
            y_pred: pd.Series,
            y_test: pd.Series
    ) -> dict:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"MSE": mse, "MAE": mae, "R2": r2}