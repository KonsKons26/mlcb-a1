import os

import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    ElasticNet, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, r_regression, f_regression
)
from sklearn.utils import resample

from mrmr import mrmr_regression

from joblib import dump, load


class Regresssor:

    VALID_MODEL_TYPES = {
        "ElasticNet",
        "SVR",
        "BayesianRidge"
    }

    VALID_METRICS ={
        "RMSE",
        "MAE",
        "R2"
    }

    VALID_MODELS = {
        "baseline",
        "feature_selection",
        "tuning"
    }

    FEATURE_SELECTION_METHODS = {
        "VarianceThreshold",
        "SelectKBest-rRegression",
        "SelectKBest-fRegression",
        "mRMR"
    }        



    def __init__(
            self,
            model_type: str,
            dataset: pd.DataFrame,
            target_col: str,
            used_metrics: list,
            models_dir: str,
            random_state: int = 42
        ):

        assert isinstance(model_type, str), "'model_type' must be <str>"
        assert (
            model_type in self.VALID_MODEL_TYPES
        ), f"'model_type' must be one of the following:\n{self.VALID_MODEL_TYPES}"
        assert isinstance(dataset, pd.DataFrame), "'dataset' must be <pd.DataFrame>"
        assert isinstance(target_col, str), "'target_col' must be <str>"
        assert isinstance(used_metrics, list), "'metrics' must be <list>"
        assert all(
            isinstance(metric, str) for metric in used_metrics
        ), "'used_metrics' must be <list> of <str>"
        assert all(
            metric in self.VALID_METRICS for metric in used_metrics
        ), f"'used_metrics' must be one of the following:\n{self.VALID_METRICS}"
        assert os.path.isdir(models_dir), "'models_dir' does not exist"
        assert isinstance(random_state, int), "'random_state' must be <int>"

        self.model_type = model_type
        self.X = dataset.drop(columns=target_col)
        self.y = dataset[target_col]
        self.used_metrics = used_metrics
        self.models_dir = models_dir
        self.random_state = random_state


    def train(
            self,
            mode: str,
            test_size: float = 0.2,
            scale_data: bool = True,
            save_model:bool = False,
            feature_selection_method: str = None,
            feature_selection_args: dict = None
        ) -> dict:
        """Trains the model based on the specified mode and evaluates its
        performance.

        Parameters
        ----------
        mode : str
            The mode of training. Must be one of the valid modes defined in
            `self.VALID_MODELS`.
        test_size : float, defaul = 0.2
            The proportion of the dataset to include in the test split.
        scale_data : bool, default = True
            Whether to scale the input data using StandardScaler.
        save_model : bool, default = False
            Whether to save the trained model and scaler (if used).
        feature_selection_method : str, optional
            The feature selection method to use. Must be one of the valid methods
            defined in `self.FEATURE_SELECTION_METHODS`.
        feature_selection_args : dict, optional
            Additional arguments for the feature selection method.

        Returns
        -------
        dict
            A dictionary containing regression metrics such as mean squared error,
            mean absolute error,  and R-squared score for the trained model on the
            test set.
        """

        if mode not in self.VALID_MODELS:
            raise ValueError(
                f"Mode must be one of the following:\n{self.VALID_MODELS}"
            )

        # Implementing a baseline model   
        if mode == "baseline":
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
                        self.models_dir,
                        f"{self.model_type}_{mode}_scaler.joblib"
                    ))

            self.model = model

            return self._regression_metrics(y_pred, y_test)
        
        # Implementing feature selection
        elif mode == "feature_selection":
            model_dict = {
                "ElasticNet": ElasticNet(random_state=self.random_state),
                "SVR": SVR(),
                "BayesianRidge": BayesianRidge()
            }
            model = model_dict[self.model_type]

            if feature_selection_method == "VarianceThreshold":
                threshold = feature_selection_args.get("threshold", 0.1)
                selector = VarianceThreshold(threshold=threshold)

            elif feature_selection_method == "SelectKBest-rRegression":
                k = feature_selection_args.get("k", 10)
                selector = SelectKBest(score_func=r_regression, k=k)

            elif feature_selection_method == "SelectKBest-fRegression":
                k = feature_selection_args.get("k", 10)
                selector = SelectKBest(score_func=f_regression, k=k)

            elif feature_selection_method == "mRMR":
                k = feature_selection_args.get("k", 10)

            else:
                raise ValueError(
                    f"feature_selection_method must be one of the following:\n{
                        self.FEATURE_SELECTION_METHODS
                    }"
                )

            all_metrics = {"RMSE": [], "MAE": [], "R2": []}

            # KFold CV
            folds = feature_selection_args.get("folds", 10)
            kf = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)

            for train_idx, test_idx in kf.split(self.X):
                X_train = self.X.iloc[train_idx]
                y_train = self.y.iloc[train_idx]
                X_test = self.X.iloc[test_idx]
                y_test = self.y.iloc[test_idx]
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
                # else:
                #     scaler = None
                #     X_train = pd.DataFrame(X_train, columns=X_train.columns)
                #     X_test = pd.DataFrame(X_test, columns=X_test.columns)
                if feature_selection_method == "mRMR":
                    X_train = mrmr_regression(X_train, y_train, k=k)
                    X_test = mrmr_regression(X_test, y_test, k=k)
                else:
                    X_train = selector.fit_transform(X_train, y_train)
                    X_test = selector.transform(X_test)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = self._regression_metrics(y_pred, y_test)
                for metric, value in metrics.items():
                    all_metrics[metric].append(value)
            if save_model:
                dump(model, os.path.join(
                    self.models_dir,
                    f"{self.model_type}_{mode}.joblib"
                ))
                if scale_data:
                    dump(scaler, os.path.join(
                        self.models_dir,
                        f"{self.model_type}_{mode}_scaler.joblib"
                    ))
            self.model = model

            # Save best feature names



            return all_metrics


    def evaluate(
            self,
            load_model: bool,
            evaluation_df: pd.DataFrame,
            target_col: str,
            model_name: str = None,
            scaled: bool = True,
            n_bootstraps: int = 100
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
        n_bootstraps : int, default = 100
            The number of bootstrap samples to use for evaluation.

        Returns
        -------
        dict
            A dictionary containing regression metrics such as mean squared error,
            mean absolute error, and R-squared score for the model on the evaluation
            data.
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

        all_metrics = {"RMSE": [], "MAE": [], "R2": []}

        for _ in range(n_bootstraps):
            X_sample, y_sample = resample(X_val, y_val)
            y_pred = model.predict(X_sample)
            metrics = self._regression_metrics(y_pred, y_sample)
            for metric, value in metrics.items():
                all_metrics[metric].append(value)

        return all_metrics


    def _regression_metrics(
            self,
            y_pred: pd.Series,
            y_test: pd.Series
    ) -> dict:
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}