from regression import Regressor

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution, FloatDistribution, CategoricalDistribution
)

class RegressorOptuna(Regressor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.study = None

    def _train_ElasticNet(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            if not tune_dict.get("param_grid"):
                param_grid = {
                    "alpha": FloatDistribution(0.01, 1.0),
                    "l1_ratio": FloatDistribution(0.0, 1.0),
                    "tol": CategoricalDistribution([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
                }
            else:
                param_grid = tune_dict.get("param_grid")
            n_folds = tune_dict.get("n_folds", 5)
            n_trials = tune_dict.get("n_trials", 100)
            timeout = tune_dict.get("timeout", None)
            self._tune_model(param_grid, n_folds, n_trials, timeout)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _train_SVR(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            if not tune_dict.get("param_grid"):
                param_grid = {
                    "kernel": CategoricalDistribution(
                        ["rbf", "linear", "poly", "sigmoid"]
                    ),
                    "degree": IntDistribution(2, 5),
                    "gamma": CategoricalDistribution(["scale", "auto"]),
                    "coef0": FloatDistribution(0.0, 1),
                    "C": FloatDistribution(0.1, 10.0),
                    "epsilon": FloatDistribution(0.0, 1.0),
                    "tol": CategoricalDistribution([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
                }
            else:
                param_grid = tune_dict.get("param_grid")
            n_folds = tune_dict.get("n_folds", 5)
            n_trials = tune_dict.get("n_trials", 100)
            timeout = tune_dict.get("timeout", None)
            self._tune_model(param_grid, n_folds, n_trials, timeout)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _train_BayesianRidge(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            if not tune_dict.get("param_grid"):
                param_grid = {
                    "alpha_1": FloatDistribution(1e-9, 1e-3, log=True),
                    "alpha_2": FloatDistribution(1e-9, 1e-3, log=True),
                    "lambda_1": FloatDistribution(1e-9, 1e-3, log=True),
                    "lambda_2": FloatDistribution(1e-9, 1e-3, log=True),
                    "tol": CategoricalDistribution([1e-3, 1e-4, 1e-5, 1e-6, 1e-7]),
                    "compute_score": CategoricalDistribution([True, False])
                }
            else:
                param_grid = tune_dict.get("param_grid")
            n_folds = tune_dict.get("n_folds", 5)
            n_trials = tune_dict.get("n_trials", 100)
            timeout = tune_dict.get("timeout", None)
            self._tune_model(param_grid, n_folds, n_trials, timeout)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _tune_model(self, param_grid, n_folds, n_trials, timeout=None):
        best_features = self._load_features(self.model_type)
        X = self.X[best_features]
        y = self.y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
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

        def objective(trial):
            # Create model with trial parameters
            params = {}
            for param_name, param_dist in param_grid.items():
                if isinstance(param_dist, IntDistribution):
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_dist.low, 
                        param_dist.high
                    )
                elif isinstance(param_dist, FloatDistribution):
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_dist.low, 
                        param_dist.high, 
                        log=getattr(param_dist, 'log', False)
                    )
                elif isinstance(param_dist, CategoricalDistribution):
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_dist.choices
                    )

            model_instance = type(self.model)(**params)

            kf = KFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )

            scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model_instance.fit(X_fold_train, y_fold_train)

                y_fold_pred = model_instance.predict(X_fold_val)

                score = r2_score(y_fold_val, y_fold_pred)
                scores.append(score)

            return np.mean(scores)

        sampler = TPESampler(seed=self.random_state)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout, 
            show_progress_bar=True
        )

        best_params = self.study.best_params
        best_model = type(self.model)(**best_params)
        best_model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_test)
        
        self.model = best_model
        self.metrics = self._regression_metrics(y_test, y_pred)
        self.metrics["best_hyperparameters"] = best_params
        self.metrics["best_score"] = self.study.best_value
        self.metrics["features"] = best_features
        self.metrics["optimization_history"] = self.study.trials_dataframe()

        self.mode = "optuna"

        return self.metrics



if __name__ == "__main__":
    model = RegressorOptuna(
        model_type="SVR",
        models_dir="/home/cotsios/dsit/2nd-semester/ml-in-comp-bio/Assignment-1/models",
        dataset=pd.read_csv("/home/cotsios/dsit/2nd-semester/ml-in-comp-bio/Assignment-1/data/development_final.csv"),
        target="BMI",
    )
    metrics = model.train(mode="tune")