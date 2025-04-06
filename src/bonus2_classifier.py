import os

import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, r_regression, f_regression
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import (
    IntDistribution, FloatDistribution, CategoricalDistribution
)

from joblib import dump, load


class Classifier:

    def __init__(
            self,
            model_type: str,
            models_dir: str,
            dataset: pd.DataFrame,
            target: str,
            random_state: int = 42
        ):

        self.model_type = model_type
        self.models_dir = models_dir
        if dataset is not None and target is not None:
            self.X = dataset.drop(columns=[target])
            self.y = dataset[target]
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.selection = None
        self.study = None
        self.metrics = None

    def train(
            self,
            mode: str = "baseline",
            feature_selection_dict: dict = {},
            tune_dict: dict = {},
            save_model: bool = True
        ) -> dict:
        self.model_name = f"{self.model_type}_{mode}"

        # LogisticRegression
        if self.model_type == "LogisticRegression":
            self.model = LogisticRegression()
            self._train_LogisticRegression(
                mode=mode,
                feature_selection_dict=feature_selection_dict,
                tune_dict=tune_dict
            )

            if save_model:
                self._save_model(self.model_name)
                self._save_scaler(self.model_name)

        # GaussianNaiveBayes
        if self.model_type == "GaussianNB":
            self.model = GaussianNB()
            self._train_GaussianNB(
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
        if mode == "baseline":
            self._train_baseline()

        if mode == "feature_selection":
            self._feature_selection(feature_selection_dict)

    def _train_baseline(self) -> None:
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


    def _tune_model(self, param_grid, n_trials, timeout=None):
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

            #
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

            return root_mean_squared_error(y_test, y_pred)

        sampler = TPESampler(seed=self.random_state)

        self.study = optuna.create_study(
            direction="minimize",
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

    def _train_LogisticRegression(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            if not tune_dict.get("param_grid"):
                param_grid = {
                    "penalty": CategoricalDistribution(["l1", "l2", "elasticnet"]),
                    "tol": FloatDistribution(1e-7, 1e-1, log=True),
                    "C": FloatDistribution(0.001, 1000, log=True),
                    "fit_intercept": CategoricalDistribution([True, False]),
                    "solver": CategoricalDistribution(
                        ["newton-cholesky", "lbfgs", "liblinear"]
                    )
                }
            else:
                param_grid = tune_dict.get("param_grid")
            n_folds = tune_dict.get("n_folds", 5)
            n_trials = tune_dict.get("n_trials", 1000)
            timeout = tune_dict.get("timeout", None)
            self._tune_model(param_grid, n_trials, timeout)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _train_GaussianNB(self, mode, feature_selection_dict, tune_dict):
        if mode == "tune":
            if not tune_dict.get("param_grid"):
                param_grid = {
                    "var_smoothing": FloatDistribution(1e-10, 1e-1, log=True),
                    "priors": CategoricalDistribution([None, [0.5, 0.5]])
                }
            else:
                param_grid = tune_dict.get("param_grid")
            n_folds = tune_dict.get("n_folds", 5)
            n_trials = tune_dict.get("n_trials", 1000)
            timeout = tune_dict.get("timeout", None)
            self._tune_model(param_grid, n_trials, timeout)
        else:
            self._train_model_no_tune(mode, feature_selection_dict)

    def _get_top_feature_selection_method(self) -> str:
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
        return {
            "RMSE": [root_mean_squared_error(y_test, y_pred)],
            "MAE": [mean_absolute_error(y_test, y_pred)],
            "R2": [r2_score(y_test, y_pred)]
        }

    def _save_model(self, model_name: str):
        model_name = model_name + ".joblib"
        dump(self.model, os.path.join(self.models_dir, model_name))

    def _save_scaler(self, scaler_name: str):
        scaler_name = scaler_name + "_scaler.joblib"
        dump(self.scaler, os.path.join(self.models_dir, scaler_name))

    def _save_features(self, model_name: str):
        features_name = model_name + "_features.txt"
        with open(os.path.join(self.models_dir, features_name), "w") as f:
            for feature in self.metrics["features"]:
                f.write(f"{feature}\n")

    def _load_model(self, model_name: str) -> object:
        model_name = model_name + ".joblib"
        model = load(os.path.join(self.models_dir, model_name))
        return model

    def _load_scaler(self, scaler_name: str) -> object:
        scaler_name = scaler_name + "_scaler.joblib"
        scaler = load(os.path.join(self.models_dir, scaler_name))
        return scaler

    def _load_features(self, model_name: str) -> list:
        features_name = f"{model_name}_features.txt"
        with open(os.path.join(self.models_dir, features_name), "r") as f:
            features = f.readlines()
        features = [feature.strip() for feature in features]
        return features


def pipeline(
      models_dir,
      dataset_dir,
      target_col,
      binary_limit=25,
      model_types=["LogisticRegression", "GaussianNB"],
      modes=["baseline", "feature_selection", "tune"],
      save_model=True  
    ):

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

    # Convert target column to binary
    binary_func = lambda x: 1 if x >= binary_limit else 0

    development_dataset[target_col] = development_dataset[
        target_col
    ].apply(binary_func)
    validation_dataset[target_col] = validation_dataset[
        target_col
    ].apply(binary_func)

    all_training_metrics = {}
    all_validation_metrics = {}

    for model_type in model_types:
        print("\n--------------------")
        print(f"{model_type:^20}")

        all_training_metrics[model_type] = {}
        all_validation_metrics[model_type] = {}

        classifier = Classifier(
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

            classifier.train(mode=mode, save_model=save_model)
            all_training_metrics[model_type][mode] = classifier.metrics

            print("Training completed")

            print("Validating... ", end="")
            metrics = classifier.validate(
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
