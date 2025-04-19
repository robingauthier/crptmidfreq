import pandas as pd
import numpy as np
import lightgbm as lgb
from crptmidfreq.utils.log import get_logger
from pprint import pprint
import optuna
from sklearn.base import BaseEstimator, RegressorMixin
log = get_logger()

lgb_params = {
    # — core boosting and loss
    'boosting_type':       'gbdt',
    # 'objective':           'tweedie',            # good for non‐negative, skewed sales
    # 'tweedie_variance_power': 1.1,                # often ~1.1 in M5‐style contests
    'objective': 'regression_l2',
    'metric':              'rmse',               # track RMSE on hold‐out

    'linear_tree': True,

    # — capacity / interaction constraints
    'num_leaves':          2**8 - 1,             # ~255 leaves; controls tree complexity
    'max_depth':           10,                   # cap depth to avoid overfitting
    'min_data_in_leaf':    100,                  # require 100 samples per leaf

    # — regularization
    'feature_fraction':    0.8,                  # randomly select 80% of features each tree
    'bagging_fraction':    0.8,                  # bag 80% of data per iteration
    'bagging_freq':        5,                    # perform bagging every 5 rounds
    'lambda_l1':           0.1,                  # L1 regularization
    'lambda_l2':           0.1,                  # L2 regularization

    # — learning rate & early stopping
    'learning_rate':       0.1,                 # extremely important to not put 1e-4
    'n_estimators':        5_000,                 # large cap; will stop early
    'early_stopping_rounds': 100,                # if no gain in X rounds → stop

    # — data bucketing & performance
    'max_bin':             100,                  # finer splits for numeric
    'subsample':           0.8,                  # equivalent to bagging_fraction
    'subsample_freq':      1,                    # subsample every iteration

    # — reproducibility & verbosity
    'seed':                42,
    'verbose': -1,
}


def check_cols(df, wcols):
    for col in wcols:
        if not col in df.columns:
            print(f'Missing col {col} in dataframe')
            assert False


class LGBMModel(BaseEstimator, RegressorMixin):
    """Includes early stopping"""

    def __init__(self, params=None, cat_features=[], num_features=None, verbose=1):
        self.params = params if params else lgb_params

        self.model = None
        self.cat_features = cat_features
        self.num_features = num_features
        self.verbose = verbose

    @staticmethod
    def preprocess_data(df, cat_features):
        ncat_features = []
        for cat in cat_features:
            if df[cat].dtype != np.int64:
                df[f'{cat}_code'] = pd.Categorical(df[cat]).codes
                df[f'{cat}_code'] = df[f'{cat}_code'].astype(np.int64)
                ncat_features += [f'{cat}_code']
            else:
                ncat_features += [cat]
        return df, ncat_features

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

        train_data = df[df['dtsi'] <= train_end]
        valid_data = df[df['dtsi'] >= train_end]

        train_dataset = lgb.Dataset(
            data=train_data[feats],
            label=train_data[target_col],
            weight=train_data[wgt_col],
            categorical_feature=self.cat_features,
        )
        valid_dataset = lgb.Dataset(
            data=valid_data[feats],
            label=valid_data[target_col],
            weight=valid_data[wgt_col],
            categorical_feature=self.cat_features,
        )
        estimator = lgb.train(lgb_params,
                              train_dataset,
                              valid_sets=[valid_dataset],
                              )

        featimp = pd.DataFrame({'name': estimator.feature_name(),
                                'imp': estimator.feature_importance()})\
            .sort_values('imp', ascending=False)
        if self.verbose > 0:
            print(featimp.head(25))
        self.model = estimator
        return self

    def predict(self, df):
        #df['ypred'] = estimator.predict(df[feats+cat_features])
        feats = self.num_features + self.cat_features
        check_cols(df, feats)
        return self.model.predict(df[feats])
