import pandas as pd
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import lightgbm as lgb
from crptmidfreq.season.holswrap import event_distances
from crptmidfreq.season.yoy import deseasonalize_yoy
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import to_csv
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.utils.log import get_logger
import pickle
from crptmidfreq.season.m5_data import get_data
from crptmidfreq.season.m5_data import dump_solution
from crptmidfreq.season.m5_data import evaluate_model

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
# a disaster is created by above
# lgb_params = {
#    'boosting_type':       'gbdt',#
#    'objective': 'regression_l2',
# }
if False:
    lgb_params = {
        'boosting_type':       'gbdt',
        'objective': 'regression_l2',
        'learning_rate':       1e-3,
        'n_estimators':        500,
    }


def fit_lgbm_model(df,
                   fd,
                   target_col='target_sales_log_vs_ewm',
                   wgt_col='wgt'):
    """
        f2 = {
        'categorial': ['c1','c2','c3'],
        'numerical': ['f1','f2','f3'],
    }
    """
    feats = list(set(fd['numerical']))

    cat_features = fd['categorical']
    ncat_features = []
    for cat in cat_features:
        if df[cat].dtype != np.int64:
            df[f'{cat}_code'] = pd.Categorical(df[cat]).codes
            df[f'{cat}_code'] = df[f'{cat}_code'].astype(np.int64)
            ncat_features += [f'{cat}_code']
        else:
            ncat_features += [cat]
    cat_features = ncat_features

    min_dt = df['date'].min()
    max_dt = df['date'].max()
    nbdays = (max_dt-min_dt).days

    train_end = min_dt+pd.to_timedelta(nbdays*0.7, 'd')
    valid_end = min_dt+pd.to_timedelta(nbdays*0.8, 'd')

    log.info(f'training end {train_end}')
    log.info(f'validation end {valid_end}')

    train_data = df[df['date'] <= train_end]
    valid_data = df[(df['date'] <= valid_end) & (df['date'] >= train_end)]

    train_dataset = lgb.Dataset(
        data=train_data[feats+cat_features],
        label=train_data[target_col],
        weight=train_data[wgt_col],
        categorical_feature=cat_features,
    )
    valid_dataset = lgb.Dataset(
        data=valid_data[feats+cat_features],
        label=valid_data[target_col],
        weight=valid_data[wgt_col],
        categorical_feature=cat_features,
    )
    log.info('training lgbm')
    estimator = lgb.train(lgb_params,
                          train_dataset,
                          valid_sets=[valid_dataset],
                          )

    featimp = pd.DataFrame({'name': estimator.feature_name(),
                            'imp': estimator.feature_importance()}).sort_values('imp', ascending=False)
    print(featimp.head(25))

    # Prediction
    log.info('predicting lgbm')
    df['ypred'] = estimator.predict(df[feats+cat_features])

    return df, {'test_start': valid_end}
