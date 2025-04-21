from pprint import pprint

import optuna
import sklearn.metrics
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np
from crptmidfreq.mllib.feedforward_sklearn import FeedForwardRegressor
from crptmidfreq.utils.log import get_logger
from crptmidfreq.mllib.nbeats2_sklearn import NBeatsNet
log = get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_optuna_ff_params(trial):
    """
    Generate a set of hyperparameters for LightGBM using Optuna.
    """
    # 1) sample number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 4)

    # 2) for each layer, sample its width from a shortlist
    hidden_units = []
    for i in range(n_layers):
        size = trial.suggest_categorical(
            f"hidden_units_layer_{i}", [32, 64, 128, 256, 512]
        )
        hidden_units.append(size)
    hidden_units = tuple(hidden_units)

    optuna_lgb_params = {
        "hidden_units": hidden_units,
        # optimizer & training
        "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        # sample each Adam beta and pack into a tuple
        "max_epochs": trial.suggest_int("max_epochs", 10, 100),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }
    return optuna_lgb_params


class FeedForwardRegressorOptuna(BaseEstimator, RegressorMixin):
    """the fit performs the optuna model selection"""

    def __init__(self,
                 input_dim,
                 n_trials=20):
        self.input_dim = input_dim
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
            model = FeedForwardRegressor(
                input_dim=self.input_dim,
                **params,
            )
            model.fit(train_X, train_y, **fit_params)
            ypred = model.predict(X)
            return ypred

        def objective(trial):
            param = get_optuna_ff_params(trial)
            ypred = fit_model(param)
            if pd.isna(ypred).mean() > 0.6:
                print('Issue of nan in NBEats2 optuna ')
                return np.inf
            test_ypred = ypred.iloc[train_stop:]
            error = sklearn.metrics.mean_absolute_error(test_ypred, test_y)
            return error
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        pprint(trial.params)
        n_layers = trial.params['n_layers']
        trial.params.pop('n_layers')
        hidden_units = []
        for i in range(n_layers):
            size = trial.params[f"hidden_units_layer_{i}"]
            trial.params.pop(f"hidden_units_layer_{i}")
            hidden_units.append(size)
        hidden_units = tuple(hidden_units)
        trial.params['hidden_units'] = hidden_units

        # fit the model with the best params
        model = FeedForwardRegressor(
            input_dim=self.input_dim,
            **trial.params,
        )
        model.fit(X, y, **fit_params)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)
