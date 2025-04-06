from src.regression import Regressor

import os

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
        optuna.logging.set_verbosity(optuna.logging.WARNING)
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

        return self.metrics


def pipeline_optuna(
        models_dir: str,
        dataset_dir: str,
        target_col: str = "BMI",
        model_types: list = ["ElasticNet", "SVR", "BayesianRidge"],
        modes: list = ["tune"],
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

        regressor = RegressorOptuna(
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