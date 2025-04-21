from pprint import pprint

import optuna
import sklearn.metrics
from sklearn.base import BaseEstimator, RegressorMixin

from crptmidfreq.mllib.lgbm_sklearn import LGBMModel, check_cols, lgb_params
from crptmidfreq.utils.log import get_logger
from crptmidfreq.mllib.nbeats2_sklearn import NBeatsNet
log = get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_optuna_nbeats2_params(trial):
    """
    Generate a set of hyperparameters for LightGBM using Optuna.
    """

    optuna_lgb_params = {

        # stack sizes
        "trend_blocks": trial.suggest_int("trend_blocks", 1, 5),
        "trend_layers": trial.suggest_int("trend_layers", 1, 4),
        "trend_layer_size": trial.suggest_categorical(
            "trend_layer_size", [20, 64, 128]
        ),
        "degree_of_polynomial": trial.suggest_int("degree_of_polynomial", 1, 5),

        "seasonality_blocks": trial.suggest_int("seasonality_blocks", 1, 5),
        "seasonality_layers": trial.suggest_int("seasonality_layers", 1, 4),
        "seasonality_layer_size": trial.suggest_categorical(
            "seasonality_layer_size", [64, 128, 256, 512]
        ),
        "num_of_harmonics": trial.suggest_int("num_of_harmonics", 1, 20),

        "generic_blocks": trial.suggest_int("generic_blocks", 0, 3),
        "generic_layers": trial.suggest_int("generic_layers", 1, 4),
        "generic_layer_size": trial.suggest_categorical(
            "generic_layer_size", [64, 128, 256, 512]
        ),

        # optimizer & training
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        # sample each Adam beta and pack into a tuple
        "max_epochs": trial.suggest_int("max_epochs", 10, 100),
        "batch_size": trial.suggest_categorical(
            "batch_size", [16, 32, 64, 128]
        ),
    }
    return optuna_lgb_params


class NBeatsNetOptuna(BaseEstimator, RegressorMixin):
    """the fit performs the optuna model selection"""

    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 n_trials=20):
        self.input_size = input_size
        self.output_size = output_size
        self.model = None
        self.n_trials = n_trials

    def fit(self, X, y=None, **fit_params):
        n = X.shape[0]

        # Split train, validation ... we are assuming this is sorted by time
        train_stop = int(n * 0.8)
        train_X = X.iloc[:train_stop]
        train_y = y.iloc[:train_stop]

        test_y = y.iloc[train_stop:]

        def fit_model(params):
            model = NBeatsNet(
                input_size=self.input_size,
                output_size=self.output_size,
                **params,
            )
            model.fit(train_X, train_y, **fit_params)
            ypred = model.predict(X)
            return ypred

        def objective(trial):
            param = get_optuna_nbeats2_params(trial)
            ypred = fit_model(param)
            test_ypred = ypred.iloc[train_stop:]
            error = sklearn.metrics.mean_absolute_error(test_ypred, test_y)
            return error
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        pprint(trial.params)

        # fit the model with the best params
        model = NBeatsNet(
            input_size=self.input_size,
            output_size=self.output_size,
            **trial.params,
        )
        model.fit(X, y, **fit_params)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)
