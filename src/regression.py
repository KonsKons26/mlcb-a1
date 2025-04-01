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

from joblib import dump, load


class Regressor:


    def __init__(
            self,
            model_type: str,
            dataset: pd.DataFrame,
            target: str,
            models_dir: str,
            random_state: int = 42
        ):

        self.model_type = model_type
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
            tune_dict: dict = {}
        ):

        model_name = f"{self.model_type}_{mode}"
        scaler_name = model_name + "_scaler.joblib"
        model_name = model_name + ".joblib"

        # ElasticNet
        if self.model_type == "ElasticNet":
            self.model = ElasticNet()
            self._train_ElasticNet(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            self._save_model(model_name)
            self._save_model(scaler_name, scaler=True)

        # SVR
        if self.model_type == "SVR":
            self.model = SVR()
            self._train_SVR(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            self._save_model(model_name)
            self._save_model(scaler_name, scaler=True)

        # BayesianRidge
        if self.model_type == "BayesianRidge":
            self.model = BayesianRidge()
            self._train_BayesianRidge(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            self._save_model(model_name)
            self._save_model(scaler_name, scaler=True)

        return self.metrics


    def evaluate(
            self,
            model_name: str,
            evalu_df: pd.DataFrame,
            target: str,
            n_bootstrap: int = 1000
        ):
        return


    def _train_model_no_tune(self, mode, feature_selection_dict):
        if mode == "baseline":
            self._train_baseline()

        if mode == "feature_selection":
            self._feature_selection(feature_selection_dict)


    def _train_ElasticNet(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            self._tune_model(tune_dict)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)


    def _train_SVR(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            self._tune_model(tune_dict)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)


    def _train_BayesianRidge(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            self._tune_model(tune_dict)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)


    def _train_baseline(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=self.random_state
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = self._regression_metrics(y_test, y_pred)

        self.metrics = metrics
        self.scaler = scaler


    def _feature_selection(self, feature_selection_dict):
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
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

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

        # Train model with the best feature selector
        best_method = self._get_top_feature_selection_method()
        selector = selectors[best_method]
        selector.fit(self.X, self.y)
        X_selected = selector.transform(self.X)
        y_selected = self.y

        self.selector = selector

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y_selected,
            test_size=0.2,
            random_state=self.random_state
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        self.scaler = scaler

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        metrics = self._regression_metrics(y_test, y_pred)
        self.metrics = metrics
        self.metrics["feature_idxs"] = selector.get_support(indices=True)
        self.metrics["features"] = self.X.columns[
            selector.get_support(indices=True)
        ].to_numpy()
        self.metrics["feature_selection_method"] = best_method


    def _get_top_feature_selection_method(self):
        # Select the method that yields the largest 
        # mean(R2) / ((mean(RMSE) + mean(MAE)) / 2)
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


    def _tune_model(self, tune_dict):
        return


    def _regression_metrics(self, y_test, y_pred):
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": [rmse], "MAE": [mae], "R2": [r2]}


    def _save_model(self, model_name, scaler=False):
        if scaler:
            dump(self.scaler, os.path.join(self.models_dir, model_name))
        else:
            dump(self.model, os.path.join(self.models_dir, model_name))
        


    def _load_model(self, model_name):
        model = load(os.path.join(self.models_dir, model_name))
        return model