import os

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, r_regression, f_regression
)
from sklearn.utils import resample

from joblib import dump, load


class Regressor:

    def __init__(
            self,
            model_type: str,
            models_dir: str,
            dataset: pd.DataFrame = None,
            target: str = None,
            random_state: int = 42
        ):

        self.model_type = model_type
        if dataset is not None and target is not None:
            self.X = dataset.drop(columns=[target])
            self.y = dataset[target]
        self.models_dir = models_dir
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.selection = None
        self.metrics = None

    def train(
            self,
            mode: str = "baseline",
            feature_selection_dict: dict = {},
            tune_dict: dict = {},
            save_model: bool = True
        ) -> dict:
        """Trains the model using the specified mode and parameters.

        Based on the model_type and mode, it will call the appropriate private
        methods, acting basically as a controller.
        The modes are:
            - baseline: Train the model without feature selection or tuning.
            - feature_selection: Perform feature selection and train the model.
            - tune: Perform hyperparameter tuning and train the model, using the
            features selected in the feature selection step.
        The feature selection and tuning dictionaries can be used to pass
        parameters to the feature selection and tuning methods, respectively; if
        they are not provided, default values will be used (recommended).

        Parameters
        ----------
        mode : str
            The mode to use for training. Options are "baseline",
            "feature_selection", or "tune".
        feature_selection_dict : dict
            A dictionary containing parameters for feature selection.
        tune_dict : dict
            A dictionary containing parameters for tuning.
        save_model : bool
            Whether to save the model and scaler after training.

        Returns
        -------
        dict
            A dictionary containing the training metrics.
        """

        self.model_name = f"{self.model_type}_{mode}"

        # ElasticNet
        if self.model_type == "ElasticNet":
            self.model = ElasticNet()
            self._train_ElasticNet(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            if save_model:
                self._save_model(self.model_name)
                self._save_scaler(self.model_name)

        # SVR
        if self.model_type == "SVR":
            self.model = SVR()
            self._train_SVR(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            if save_model:
                self._save_model(self.model_name)
                self._save_scaler(self.model_name)

        # BayesianRidge
        if self.model_type == "BayesianRidge":
            self.model = BayesianRidge()
            self._train_BayesianRidge(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            if save_model:
                self._save_model(self.model_name)
                self._save_scaler(self.model_name)

        return self.metrics

    def validate(
            self,
            model_name: str,
            mode: str,
            val_df: pd.DataFrame,
            target: str,
            n_bootstrap: int = 1000
        ) -> dict:
        """Validates the model using the specified mode and parameters.

        Validation is performed using bootstrap sampling, which allows
        estimating the performance of the model on unseen data. The
        validation metrics are calculated for each bootstrap sample.

        Parameters
        ----------
        model_name : str
            The name of the model to validate.
        mode : str
            The mode to use for validation. Options are "baseline",
            "feature_selection", or "tune".
        val_df : pd.DataFrame
            The validation dataset to use for validation.
        target : str
            The target variable to predict.
        n_bootstrap : int
            The number of bootstrap samples to use for validation.

        Returns
        -------
        dict
            A dictionary containing the validation metrics.        

        """

        self.model_name = f"{model_name}_{mode}"

        X_val = val_df.drop(columns=[target])
        y_val = val_df[target]

        model = self._load_model(self.model_name)
        scaler = self._load_scaler(self.model_name)
        if mode != "baseline":
            best_features = self._load_features(self.model_name.split("_")[0])
            X_val = X_val[best_features]

        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        all_metrics = {"RMSE": [], "MAE": [], "R2": []}

        for _ in range(n_bootstrap):
            X_sample, y_sample = resample(X_val, y_val)

            y_pred = model.predict(X_sample)
            metrics = self._regression_metrics(y_sample, y_pred)

            for metric_name, metric_values in metrics.items():
                all_metrics[metric_name].extend(metric_values)

        return all_metrics

    def _train_model_no_tune(self, mode, feature_selection_dict) -> None:
        """Trains the model without tuning, controller private method."""
        if mode == "baseline":
            self._train_baseline()

        if mode == "feature_selection":
            self._feature_selection(feature_selection_dict)

    def _train_baseline(self) -> None:
        """Trains the model without tuning or feature selection.

        This method is used to train the model using the entire dataset
        without any feature selection or tuning. It splits the dataset into
        training and testing sets, standardizes the features, and fits
        the model to the training data. The metrics are calculated on the
        testing set and stored in the metrics attribute.
        The model is saved to the models directory.
        The scaler is also saved to the models directory.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=self.random_state
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = self._regression_metrics(y_test, y_pred)

        self.metrics = metrics
        self.scaler = scaler

    def _feature_selection(self, feature_selection_dict):
        """Performs feature selection and trains the model.

        This method is used to perform feature selection using different
        methods and train the model using the selected features. The selection
        of the feature selection method is based on the metrics calculated
        during cross-validation. The method iterates over different
        feature selection methods, applies them to the dataset, and
        calculates the metrics using KFold cross-validation. In essence, the
        feature selector is chosen based on the method that yields the best
        performance for a specific model, so this method is NOT model agnostic.
        After the best feature selector is found, it is used to transform the
        dataset and train the model. The metrics are calculated on the
        testing set and stored in the metrics attribute. The model and scaler are
        saved to the models directory. The features selected are also saved
        to the models directory.

        The default values are for the feature selection dictionary are:
            - threshold: 0.1
            - k: 20
            - n_folds: 5
        The feature selection methods used are:
            - VarianceThreshold: Removes features with variance below the
            threshold.
            - SelectKBest-r_regression: Selects the top k features based on
            the r_regression score function.
            - SelectKBest-f_regression: Selects the top k features based on
            the f_regression score function.

        Parameters
        ----------
        feature_selection_dict : dict
            A dictionary containing parameters for feature selection.
            The keys are:
                - threshold: The threshold for the VarianceThreshold method.
                - k: The number of features to select for the SelectKBest method.
                - n_folds: The number of folds for KFold cross-validation.

        """
        threshold = feature_selection_dict.get("threshold", 0.1)
        k = feature_selection_dict.get("k", 20)
        n_folds = feature_selection_dict.get("n_folds", 5)

        selectors = {
            "VarianceThreshold": VarianceThreshold(threshold=threshold),
            "SelectKBest-r_regression": SelectKBest(score_func=r_regression, k=k),
            "SelectKBest-f_regression": SelectKBest(score_func=f_regression, k=k)
        }

        all_metrics = {
            "VarianceThreshold": {},
            "SelectKBest-r_regression": {},
            "SelectKBest-f_regression": {}
        }

        # Check every selector and calculate metrics using KFold cross-validation
        for selector_name, selector in selectors.items():
            metrics_per_selector = {"RMSE": [], "MAE": [], "R2": [], "features": []}

            kf = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.random_state
            )

            for train_index, test_index in kf.split(self.X):
                # Subsets
                X_train = self.X.iloc[train_index]
                X_test = self.X.iloc[test_index]
                y_train = self.y.iloc[train_index]
                y_test = self.y.iloc[test_index]

                # Standardizartion
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(
                    scaler.transform(X_train),
                    columns=X_train.columns
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns
                )

                # Feature selection
                selector.fit(X_train, y_train)
                X_train = selector.transform(X_train)
                X_test = selector.transform(X_test)

                # Model training and evaluation
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                # Store metrics
                metrics = self._regression_metrics(y_test, y_pred)
                for metric_name, metric_values in metrics.items():
                    metrics_per_selector[metric_name].extend(metric_values)

            all_metrics[selector_name] = metrics_per_selector

        self.metrics = all_metrics

        # Train model with the best feature selector found
        best_method = self._get_top_feature_selection_method()
        selector = selectors[best_method]
        selector.fit(self.X, self.y)
        X_selected = selector.transform(self.X)
        y_selected = self.y

        cols = selector.get_support(indices=True)
        X_selected = pd.DataFrame(X_selected, columns=self.X.columns[cols])

        self.selector = selector

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y_selected,
            test_size=0.2,
            random_state=self.random_state
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_train.columns
        )

        self.scaler = scaler

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        metrics = self._regression_metrics(y_test, y_pred)
        self.metrics = metrics
        self.metrics["features"] = X_selected.columns.to_numpy()
        self.metrics["feature_idxs"] = selector.get_support(indices=True)
        self.metrics["feature_selection_method"] = best_method

        self._save_features(self.model_type)

    def _tune_model(self, param_grid, n_folds) -> None:
        """Tunes the model using GridSearchCV.

        This method is used to perform hyperparameter tuning using GridSearchCV. It
        splits the dataset into training and testing sets, standardizes the features,
        and fits the model to the training data. The metrics are calculated on the
        testing set and stored in the metrics attribute. The model is saved to the
        models directory. The scaler is also saved to the models directory.

        Parameters
        ----------
        param_grid : dict
            A dictionary containing the hyperparameters to tune.
        n_folds : int
            The number of folds for KFold cross-validation.
        """
        best_features = self._load_features(self.model_type)
        X = self.X[best_features]
        y = self.y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            columns=best_features
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=best_features
        )

        self.scaler = scaler

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
            refit="r2",
            cv=n_folds,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_hyperparameters = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.model = best_model
        y_pred = self.model.predict(X_test)

        self.metrics = self._regression_metrics(y_test, y_pred)
        self.metrics["best_hyperparameters"] = best_hyperparameters
        self.metrics["best_score"] = best_score
        self.metrics["features"] = best_features

    def _train_ElasticNet(
        self,
        mode: str,
        feature_selection_dict: dict,
        tune_dict: dict
    ) -> None:
        """Trains the ElasticNet model.

        This private method is used to train the ElasticNet model. It calls the
        appropriate private methods based on the mode. If no tuning dictionary is
        provided, default values are used, specific for this model type.

        Parameters
        ----------
        mode : str
            The mode to use for training. Options are "baseline",
            "feature_selection", or "tune".
        feature_selection_dict : dict
            A dictionary containing parameters for feature selection.
        tune_dict : dict
            A dictionary containing parameters for tuning.
        """
        if mode == "tune":
            grid_size = tune_dict.get("grid_size", 50)
            if tune_dict.get("param_grid") is None:
                param_grid = {
                    "alpha": np.linspace(0.1, 1.0, grid_size),
                    "l1_ratio": np.linspace(0.1, 1.0, grid_size),
                    "tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                }
            else:
                param_grid = tune_dict.get("param_grid", {})
            n_folds = tune_dict.get("n_folds", 5)
            self._tune_model(param_grid, n_folds)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _train_SVR(
        self,
        mode: str,
        feature_selection_dict: dict,
        tune_dict: dict
    ) -> None:
        """Trains the SVR model.

        This private method is used to train the ElasticNet model. It calls the
        appropriate private methods based on the mode. If no tuning dictionary is
        provided, default values are used, specific for this model type.

        Parameters
        ----------
        mode : str
            The mode to use for training. Options are "baseline",
            "feature_selection", or "tune".
        feature_selection_dict : dict
            A dictionary containing parameters for feature selection.
        tune_dict : dict
            A dictionary containing parameters for tuning.
        """
        if mode == "tune":
            grid_size = tune_dict.get("grid_size", 10)
            if tune_dict.get("param_grid") is None:
                param_grid = {
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "degree": [2, 3, 4],
                    "gamma": ["scale", "auto"],
                    "coef0": np.linspace(0.0, 1, grid_size),
                    "tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                    "C": np.linspace(0.1, 1.0, grid_size),
                    "epsilon": np.linspace(0.0, 1.0, grid_size)
                }
            else:
                param_grid = tune_dict.get("param_grid", {})
            n_folds = tune_dict.get("n_folds", 5)
            self._tune_model(param_grid, n_folds)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _train_BayesianRidge(
        self,
        mode: str,
        feature_selection_dict: dict,
        tune_dict: dict
    ) -> None:
        """Trains the BayesianRidge model.

        This private method is used to train the ElasticNet model. It calls the
        appropriate private methods based on the mode. If no tuning dictionary is
        provided, default values are used, specific for this model type.

        Parameters
        ----------
        mode : str
            The mode to use for training. Options are "baseline",
            "feature_selection", or "tune".
        feature_selection_dict : dict
            A dictionary containing parameters for feature selection.
        tune_dict : dict
            A dictionary containing parameters for tuning.
        """
        if mode == "tune":
            grid_size = tune_dict.get("grid_size", 10)
            if tune_dict.get("param_grid") is None:
                param_grid = {
                    "tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                    "alpha_1": np.linspace(1e-3, 1e-9, grid_size),
                    "alpha_2": np.linspace(1e-3, 1e-9, grid_size),
                    "lambda_1": np.linspace(1e-3, 1e-9, grid_size),
                    "lambda_2": np.linspace(1e-3, 1e-9, grid_size),
                    "compute_score": [True, False]
                }
            else:
                param_grid = tune_dict.get("param_grid", {})
            n_folds = tune_dict.get("n_folds", 5)
            self._tune_model(param_grid, n_folds)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _get_top_feature_selection_method(self) -> str:
        """Selects the feature selection method that yields the best performance.

        This method calculates the performance of each feature selection
        method based on the metrics calculated during cross-validation. It
        returns the name of the feature selection method that yields the
        best performance.
    
        The performance is calculated as the ratio of the mean R2 score
        to the mean RMSE and MAE scores:
        performance = mean_R2 / ((mean_RMSE + mean_MAE) / 2)

        The method with the highest performance is selected as the best feature
        selection method.
        """
        all_methods = {
            "VarianceThreshold": [],
            "SelectKBest-r_regression": [],
            "SelectKBest-f_regression": []
        }
        for selection_method, metrics in self.metrics.items():
            rmse = metrics["RMSE"]
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            mean_r2 = sum(r2) / len(r2)
            mean_rmse = sum(rmse) / len(rmse)
            mean_mae = sum(mae) / len(mae)

            all_methods[selection_method] = mean_r2 / ((mean_rmse + mean_mae) / 2)

        best_method = max(all_methods, key=all_methods.get)

        return best_method

    def _regression_metrics(self, y_test, y_pred) -> dict:
        """Calculates regression metrics.

        This method calculates the regression metrics for the model
        predictions. It calculates the RMSE, MAE, and R2 score for the
        predictions and returns them in a dictionary.
        The metrics are calculated using the sklearn.metrics module.

        Parameters
        ----------
        y_test : array-like
            The true target values.
        y_pred : array-like
            The predicted target values.

        Returns
        -------
        dict
            A dictionary containing the regression metrics. The keys are:
                - RMSE: The root mean squared error.
                - MAE: The mean absolute error.
                - R2: The R2 score.
        """
        return {
            "RMSE": [root_mean_squared_error(y_test, y_pred)],
            "MAE": [mean_absolute_error(y_test, y_pred)],
            "R2": [r2_score(y_test, y_pred)]
        }

    def _save_model(self, model_name: str):
        """Saves the model to the models directory."""
        model_name = model_name + ".joblib"
        dump(self.model, os.path.join(self.models_dir, model_name))

    def _save_scaler(self, scaler_name: str):
        """Saves the scaler to the models directory."""
        scaler_name = scaler_name + "_scaler.joblib"
        dump(self.scaler, os.path.join(self.models_dir, scaler_name))

    def _save_features(self, model_name: str):
        """Saves the features to the models directory."""
        features_name = model_name + "_features.txt"
        with open(os.path.join(self.models_dir, features_name), "w") as f:
            for feature in self.metrics["features"]:
                f.write(f"{feature}\n")

    def _load_model(self, model_name: str) -> object:
        """Loads the model from the models directory."""
        model_name = model_name + ".joblib"
        model = load(os.path.join(self.models_dir, model_name))
        return model

    def _load_scaler(self, scaler_name: str) -> object:
        """Loads the scaler from the models directory."""
        scaler_name = scaler_name + "_scaler.joblib"
        scaler = load(os.path.join(self.models_dir, scaler_name))
        return scaler

    def _load_features(self, model_name: str) -> list:
        """Loads the features from the models directory."""
        features_name = f"{model_name}_features.txt"
        with open(os.path.join(self.models_dir, features_name), "r") as f:
            features = f.readlines()
        features = [feature.strip() for feature in features]
        return features


def pipeline(
        models_dir: str,
        dataset_dir: str,
        target_col: str = "BMI",
        model_types: list = ["ElasticNet", "SVR", "BayesianRidge"],
        modes: list = ["baseline", "feature_selection", "tune"],
        save_model: bool = True
    ) -> dict:
    """Hands-off pipeline for the regression models.

    This function runs the entire pipeline for the regression models. It
    loads the datasets, trains the models, validates them, and saves the
    models and scalers. The models are trained using different modes:
    baseline, feature_selection, and tune. The models are saved to the
    models directory. The scalers are also saved to the models directory.

    Parameters
    ----------
    models_dir : str
        The directory where the models and scalers will be saved.
    dataset_dir : str
        The directory where the datasets are located.
    target_col : str
        The target variable to predict. Default is "BMI".
    model_types : list
        The list of model types to use for training. Default is
        ["ElasticNet", "SVR", "BayesianRidge"].
    modes : list
        The list of modes to use for training. Default is
        ["baseline", "feature_selection", "tune"].
    save_model : bool
        Whether to save the model and scaler after training. Default is
        True.

    Returns
    -------
    dict
        A dictionary containing the training and validation metrics for
        each model type and mode. The keys are:
            - training: A dictionary containing the training metrics.
            - validation: A dictionary containing the validation metrics.
    """

    print("----------------")
    print("Running pipeline")
    print("----------------")

    development_dataset_full_path = os.path.join(
        dataset_dir,
        "development_final.csv"
    )
    validation_dataset_full_path = os.path.join(
        dataset_dir,
        "validation_final.csv"
    )

    development_dataset = pd.read_csv(
        development_dataset_full_path,
        index_col=0
    )
    validation_dataset = pd.read_csv(
        validation_dataset_full_path,
        index_col=0
    )

    all_training_metrics = {}
    all_validation_metrics = {}

    for model_type in model_types:
        print("\n--------------------")
        print(f"{model_type:^20}")

        all_training_metrics[model_type] = {}
        all_validation_metrics[model_type] = {}

        regressor = Regressor(
            model_type=model_type,
            models_dir=models_dir,
            dataset=development_dataset,
            target=target_col
        )

        for mode in modes:
            print("--------------------")
            print(f"{mode:^20}")
            print("--------------------")
            print("Training... ", end="")
            regressor.train(mode=mode, save_model=save_model)
            metrics = regressor.metrics
            all_training_metrics[model_type][mode] = metrics
            print("Training completed")

            print("Validating... ", end="")
            metrics = regressor.validate(
                model_name=model_type,
                mode=mode,
                val_df=validation_dataset,
                target=target_col
            )
            all_validation_metrics[model_type][mode] = metrics
            print("Validation completed")

    print("\n------------------")
    print("Pipeline completed")
    print("------------------\n")
    return {"training": all_training_metrics, "validation": all_validation_metrics}


def inference(
        model_path: str,
        model_name: str,
        mode: str,
        test_df: pd.DataFrame,
        target_name: str,
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
        plot_metrics: bool = True
    ) -> dict | np.ndarray:
    """Performs inference using the trained model.

    Parameters
    ----------
    model_path : str
        The path to the directory where the model is saved.
    model_name : str
        The name of the model to use for inference. Options are
        "ElasticNet", "SVR", or "BayesianRidge".
    mode : str
        The mode to use for inference. Options are "baseline",
        "feature_selection", or "tune".
    test_df : pd.DataFrame
        The test dataset to use for inference.
    target_name : str
        The target variable to predict.
    bootstrap : bool, default=False
        Whether to use bootstrap sampling for inference.
    n_bootstrap : int, default=1000
        The number of bootstrap samples to use for inference.
    plot_metrics : bool, default=True
        Whether to plot the metrics after inference.

    Returns
    -------
    dict or np.ndarray
        If bootstrap is True, returns a dictionary containing the
        regression metrics for each bootstrap sample. If bootstrap is
        False, returns the predicted values for the test dataset.
    """

    all_metrics = {"RMSE": [], "MAE": [], "R2": []}

    model = load(os.path.join(
        model_path, f"{model_name}_{mode}.joblib"
    ))
    scaler = load(os.path.join(
        model_path, f"{model_name}_{mode}_scaler.joblib"
    ))
    with open(os.path.join(model_path, f"{model_name}_features.txt"), "r") as f:
        fs = f.readlines()
    features = [f.strip() for f in fs]

    X = test_df.drop(columns=target_name)
    y = test_df[target_name]

    X = X[features]

    X = pd.DataFrame(scaler.transform(X), columns = X.columns)

    if bootstrap:
        for _ in range(n_bootstrap):
            X_sample, y_sample = resample(X, y)

            y_pred = model.predict(X_sample)

            all_metrics["RMSE"].append(root_mean_squared_error(y_sample, y_pred))
            all_metrics["MAE"].append(mean_absolute_error(y_sample, y_pred))
            all_metrics["R2"].append(r2_score(y_sample, y_pred))

        print(f"{'metric':^20} | {'mean':^20} | {'median':^20} | {'std':^20}")
        print("-" * 89)
        for metric, values in all_metrics.items():
            mean = np.mean(values)
            median = np.median(values)
            std = np.std(values)
            print(f"{metric:^20} | {mean:^20.4f} | {median:^20.4f} | {std:^20.4f}")

        if plot_metrics:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            metrics = ["RMSE", "MAE", "R2"]
            for i, metric in enumerate(metrics):
                sns.boxplot(data=all_metrics[metric], ax=axes[i])
                axes[i].set_title(metric)
                axes[i].set_xlabel("Bootstrap samples")
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
            plt.tight_layout()
            plt.show()

        return all_metrics
 
    else:
        y_pred = model.predict(X)

        rmse = root_mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print("RMSE: ", rmse)
        print("MAE: ", mae)
        print("R2: ", r2)

        return y_pred
