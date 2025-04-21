from pprint import pprint

import optuna
import sklearn.metrics
from sklearn.base import BaseEstimator, RegressorMixin

from crptmidfreq.mllib.lgbm_sklearn import LGBMModel, check_cols, lgb_params
from crptmidfreq.utils.log import get_logger

log = get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_optuna_lgb_params(trial):
    """
    Generate a set of hyperparameters for LightGBM using Optuna.
    """

    optuna_lgb_params = {
        "objective": trial.suggest_categorical("objective", ["regression_l2", "tweedie"]),
        "metric": "rmse",
        'linear_tree': trial.suggest_categorical("linear_tree", [True, False]),
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "learning_rate": trial.suggest_float("learning_rate",  1e-4, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 2**8-1),
    }
    # pprint(optuna_lgb_params)
    return dict(optuna_lgb_params, **lgb_params)


class LGBMModelOptuna(BaseEstimator, RegressorMixin):
    """the fit performs the optuna model selection"""

    def __init__(self, cat_features=[], num_features=None, n_trials=20):
        self.cat_features = cat_features
        self.num_features = num_features
        self.model = None
        self.n_trials = n_trials

    @staticmethod
    def preprocess_data(df, cat_features):
        return LGBMModel.preprocess_data(df, cat_features)

    def fit(self, df, target_col, wgt_col='one'):
        if 'one' not in df.columns:
            df['one'] = 1

        feats = self.num_features + self.cat_features
        check_cols(df, feats)
        check_cols(df, [target_col, wgt_col, 'dtsi'])

        # Split train, validation ( for early stopping)
        min_dt = df['dtsi'].min()
        max_dt = df['dtsi'].max()
        nbdays = (max_dt-min_dt)
        train_end = min_dt+nbdays*0.8

        train_df = df[df['dtsi'] <= train_end].copy()

        def fit_lgbm_model(params):
            model = LGBMModel(params=params,
                              cat_features=self.cat_features,
                              num_features=self.num_features,
                              verbose=0)
            model.fit(train_df, target_col=target_col, wgt_col=wgt_col)
            df['ypred'] = model.predict(df[feats])
            return df

        def objective(trial):
            param = get_optuna_lgb_params(trial)
            df = fit_lgbm_model(param)
            valid_df = df.loc[lambda x:x['dtsi'] >= train_end]
            error = sklearn.metrics.mean_absolute_error(valid_df[target_col], valid_df['ypred'])
            return error
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        pprint(trial.params)

        # fit the model with the best params
        model = LGBMModel(params=trial.params, cat_features=self.cat_features, num_features=self.num_features)
        model.fit(train_df, target_col=target_col, wgt_col=wgt_col)
        self.model = model
        return self

    def predict(self, df):
        #df['ypred'] = estimator.predict(df[feats+cat_features])
        feats = self.num_features + self.cat_features
        check_cols(df, feats)
        return self.model.predict(df[feats])
