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
from crptmidfreq.season.res.kagstore.kagstore_data import get_data
from crptmidfreq.season.res.kagstore.kagstore_feature import add_features
from crptmidfreq.mllib.lgbm_sklearn_optuna import LGBMModelOptuna
from crptmidfreq.mllib.lgbm_sklearn import LGBMModel
#from crptmidfreq.season.gluonts_v1 import fit_gluonts_model
import sklearn.metrics
import optuna
# sales_log= log(1+sales)
# exp(sales_log)= 1+sales
# sales= exp(sales_log)-1

data_start = pd.to_datetime('2013-01-01')
valid_start = pd.to_datetime('2015-09-01')  # for optuna
test_start = pd.to_datetime('2016-09-01')
data_end = pd.to_datetime('2017-08-15')


def evaluate_model(df, name='', verbose=True):
    first_date = df['date'].min()
    mae = (df['sales']-df['pred_sales']).abs().mean()
    nbpoints = df.shape[0]
    if verbose:
        print(f'MAE model {name} is mae:{mae:.2f}  -- cnt:{nbpoints} -- date>={first_date}')
    return mae


def dump_data(df, name='', with_wcols=True):
    wcols = ['date', 'sales', 'pred_sales']
    fams = ['LIQUOR,WINE,BEER', 'AUTOMOTIVE', 'FROZEN FOODS']
    for fam in fams:
        if with_wcols:
            tdf = df.loc[lambda x:x['family'] == fam][wcols].sort_values('date')
        else:
            tdf = df.loc[lambda x:x['family'] == fam].sort_values('date')
        tdf['d'] = tdf['sales']-tdf['pred_sales']
        tdf['cd'] = tdf['d'].cumsum()
        to_csv(tdf, f'example_kagstore_{name}_{fam}')


def main_lgbm_optuna():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModelOptuna(cat_features=f2['categorical'], num_features=f2['numerical'], n_trials=30)
    model.fit(df_train, target_col='sales', wgt_col='one')
    df_test['pred_sales'] = model.predict(df_test)
    df['pred_sales'] = model.predict(df)

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='lgbm_optuna')
    return {'mae': mae, 'name': 'lgbm_optuna'}


def main_lgbm():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModel(cat_features=f2['categorical'], num_features=f2['numerical'])
    model.fit(df_train, target_col='sales', wgt_col='one')
    df_test['pred_sales'] = model.predict(df_test)
    df['pred_sales'] = model.predict(df)

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='lgbm')
    return {'mae': mae, 'name': 'lgbm'}


def main_lgbm_relative():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModelOptuna(cat_features=f2['categorical'], num_features=f2['numerical'], n_trials=30)
    model.fit(df_train, target_col='sales_vs_ewm', wgt_col='one')
    df_test['pred_sales_vs_ewm'] = model.predict(df_test)
    df_test['pred_sales_log'] = df_test['pred_sales_vs_ewm']+df_test['wgt']
    df_test['pred_sales'] = np.exp(df_test['pred_sales_log'])-1.0

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    return {'mae': mae, 'name': 'lgbm_relative'}


def main_lgbm_bottom_up():
    df, f1 = get_data(agglevel=None)
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()

    # dataset is too big for optuna
    model = LGBMModel(cat_features=f2['categorical'], num_features=f2['numerical'])
    model.fit(df_train, target_col='sales', wgt_col='one')
    df['pred_sales'] = model.predict(df)

    # Aggregation now
    dfg = df.groupby(['family', 'date'])\
        .agg({'sales': 'sum', 'pred_sales': 'sum', 'onpromotion': 'sum'}).reset_index()

    dfg_test = dfg.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(dfg_test, name='kaggle-store')
    dump_data(df, name='lgbm_bottom_up')
    return {'mae': mae, 'name': 'lgbm_bottom_up'}


def main_naive_ewm():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['pred_sales'] = df['wgt']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naivecst')

    return {'mae': mae, 'name': 'naive_cst'}


def main_naive_1y_lag():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)
    # we clip the YoY rate
    df['sales_vs_ewm_lag1y'] = df['sales_vs_ewm_lag1y'].clip(lower=-0.5, upper=0.5)
    df['pred_sales_log'] = df['sales_vs_ewm_lag1y']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1y', with_wcols=False)
    return {'mae': mae, 'name': 'naive_1y_lag'}


def main_naive_1y_lag_hols():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)
    # we clip the YoY rate
    df['sales_vs_ewm_lag1y'] = df['sales_vs_ewm_lag1y'].clip(lower=-0.5, upper=0.5)
    df['pred_sales_log'] = df['sales_vs_ewm_lag1y']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1yh')

    return {'mae': mae, 'name': 'naive_1y_lag_hols'}


def main_naive_1y_lag_absolute():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['pred_sales'] = df['sales_lag1y']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1y_lag_absolute')
    return {'mae': mae, 'name': 'naive_1y_lag_absolute'}


def main():
    """
            mae                   name
    2  2154.596320         lgbm_bottom_up
    0  2687.344162                   lgbm
    1  3105.981503            lgbm_optuna
    6  3616.366427  naive_1y_lag_absolute
    7  5058.685658              naive_cst
    3  5242.093887          lgbm_relative
    4  5298.429890           naive_1y_lag
    5  5298.429890      naive_1y_lag_hols
    """
    r = []
    r += [main_lgbm()]
    r += [main_lgbm_optuna()]
    r += [main_lgbm_bottom_up()]  # slow
    r += [main_lgbm_relative()]
    r += [main_naive_1y_lag()]
    r += [main_naive_1y_lag_hols()]
    r += [main_naive_1y_lag_absolute()]
    r += [main_naive_ewm()]

    rdf = pd.DataFrame(r)
    print('-'*20)
    print('-'*20)
    print(rdf.sort_values('mae'))


# ipython -i -m crptmidfreq.season.res.kagstore.kagstore_main
if __name__ == '__main__':
    # TODO add 1 model per category pls !
    # main_lgbm_relative()
    # main_naive_1y_lag()
    # main_lgbm_bottom_up()
    main()
    # main_lgbm_bottom_up()
