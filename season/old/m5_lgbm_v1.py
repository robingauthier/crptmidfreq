import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.season.m5_data import evaluate_model, get_data
from crptmidfreq.season.yoy import deseasonalize_yoy
from crptmidfreq.utils.log import get_logger

log = get_logger()

lgb_params = {
    # — core boosting and loss
    'boosting_type':       'gbdt',
    # 'objective':           'tweedie',            # good for non‐negative, skewed sales
    # 'tweedie_variance_power': 1.1,                # often ~1.1 in M5‐style contests
    'objective': 'regression_l2',
    'metric':              'rmse',               # track RMSE on hold‐out

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
    'learning_rate':       0.03,                 # slow and steady
    'n_estimators':        3000,                 # large cap; will stop early
    'early_stopping_rounds': 100,                # if no gain in 100 rounds → stop

    # — data bucketing & performance
    'max_bin':             255,                  # finer splits for numeric
    'subsample':           0.8,                  # equivalent to bagging_fraction
    'subsample_freq':      1,                    # subsample every iteration

    # — reproducibility & verbosity
    'seed':                42,
    'verbose': -1,
}
TARGET = 'sales'         # Our main target
g_folder = os.path.join(get_feature_folder(), 'm5')+'/'
os.makedirs(g_folder, exist_ok=True)


def pandas_to_dict(df):
    r = {}
    for col in df.columns:
        r[col] = df[col].values
    return r


def add_features(df):
    df['dscode_str'] = df['dept_id']+'_'+df['store_id']
    df['dscode'] = pd.Categorical(df['dscode_str']).codes

    assert 'dtsi' in df.columns
    assert 'dscode' in df.columns

    df['dscode'] = df['dscode'].astype(np.int64)
    df['dtsi'] = df['dtsi'].astype(np.int64)

    defargs = {'folder': g_folder}
    featd = pandas_to_dict(df)

    # lagging the target
    featd, lagsales = perform_lag(featd, feats=['sales_log'], windows=[1], **defargs)
    featd, lagpx = perform_lag(featd, feats=['sell_price_log'], windows=[1], **defargs)

    # This will be our target log(sales/lag(ewm(sales)))
    featd, ewmsales = perform_ewm(featd, feats=lagsales, windows=[20], **defargs)
    featd['target_sales_log_vs_ewm'] = featd[lagsales[0]]-featd[ewmsales[0]]
    featd['wgt'] = featd[ewmsales[0]]

    # Feature 1 : sales/ewm(sales)
    featd['sales_log_lag_vs_ewm'] = featd['sales_log']-featd[ewmsales[0]]

    # Feature 2 : px/ewm(px)
    featd, ewmpx = perform_ewm(featd, feats=lagpx, windows=[20], **defargs)
    featd['sell_price_log_lag_vs_ewm'] = featd[lagpx[0]]-featd[ewmpx[0]]

    # Feature 3 : lags of sales/ewm sales
    featd, ewmsaleslag = perform_lag(featd, feats=['sales_log_lag_vs_ewm'], windows=[20, 365])
    ndf = pd.DataFrame(featd)

    # Feature 4 : 1 year lag on same day
    ndf, salyoy = deseasonalize_yoy(ndf,
                                    date_col='date',
                                    stock_col='dscode',
                                    serie_col='sales_log_lag_vs_ewm')

    ndf, pxyoy = deseasonalize_yoy(ndf,
                                   date_col='date',
                                   stock_col='dscode',
                                   serie_col='sell_price_log_lag_vs_ewm')

    nfeats = []+salyoy+pxyoy

    return ndf, nfeats


def fit_model(df, feats):

    cat_features = ['weekday',
                    'wday',
                    'month',
                    'wm_yr_wk',
                    'evt_all_next',
                    'evt_all_prev',
                    'dscode',
                    'dept_id',
                    'cat_id',
                    'store_id',
                    'state_id',
                    ]
    ncat_features = []
    for cat in cat_features:
        if df[cat].dtype != np.int64:
            df[f'{cat}_code'] = pd.Categorical(df[cat]).codes
            df[f'{cat}_code'] = df[f'{cat}_code'].astype(np.int64)
            ncat_features += [f'{cat}_code']
        else:
            ncat_features += [cat]
    cat_features = ncat_features

    feats += ['dist_since_last_all',
              'dist_to_next_all',
              'dist_since_last_High',
              'dist_to_next_High'
              ]
    feats = list(set(feats))

    min_dt = df['date'].min()
    max_dt = df['date'].max()
    nbdays = (max_dt-min_dt).days

    train_end = min_dt+pd.to_timedelta(nbdays*0.7, 'd')
    valid_end = min_dt+pd.to_timedelta(nbdays*0.8, 'd')

    train_data = df[df['date'] <= train_end]
    valid_data = df[(df['date'] <= valid_end) & (df['date'] >= train_end)]

    target_col = 'target_sales_log_vs_ewm'
    wgt_col = 'wgt'

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
    df['pred_sales_log'] = df['ypred']+df['wgt']

    return df, {'test_start': valid_end}


# ipython -i -m crptmidfreq.season.m5_lgbm_v1
if __name__ == '__main__':
    if False:
        df = get_data()
        #to_csv(df.loc[lambda x:x['dept_id'] == 'HOUSEHOLD_1'], 'example_HOUSEHOLD_1')
        df, feats = add_features(df)

        pickle.dump({'df': df, 'feats': feats}, open(g_folder+'temp.pkl', 'wb'))

    fd = pickle.load(open(g_folder+'temp.pkl', 'rb'))
    df = fd['df']
    feats = fd['feats']
    df, rd = fit_model(df, feats)

    test_start = rd['test_start']
    df_test = df.loc[lambda x:x['date'] >= test_start]
    evaluate_model(df_test, 'lgbm_global')
    # MAE model lgbm_global is mae:1.13  -- cnt:27230 -- date>=2015-05-01 00:00:00
