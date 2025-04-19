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
from crptmidfreq.season.kagstore_data import get_data
from crptmidfreq.season.kagstore_data import evaluate_model
from crptmidfreq.season.kagstore_feature import add_features
from crptmidfreq.season.lgbm_v1 import fit_lgbm_model

# sales_log= log(1+sales)
# exp(sales_log)= 1+sales
# sales= exp(sales_log)-1


def dump_data(df, name=''):
    wcols = ['date', 'sales', 'pred_sales']
    fams = ['LIQUOR,WINE,BEER', 'AUTOMOTIVE', 'FROZEN FOODS']
    for fam in fams:
        tdf = df.loc[lambda x:x['family'] == fam][wcols].sort_values('date')
        tdf['d'] = tdf['sales']-tdf['pred_sales']
        tdf['cd'] = tdf['d'].cumsum()
        to_csv(tdf, f'example_kagstore_{name}_{fam}')


def main_lgbm1():
    df, f1 = get_data()
    df, f2 = add_features(df, f1)
    df, f3 = fit_lgbm_model(df, f2, target_col='target_sales_vs_ewm', wgt_col='wgt')
    df['pred_sales_log'] = df['ypred']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    test_start = f3['test_start']
    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='lgbm1')
    return {'mae': mae, 'name': 'lgbm1'}


def main_lgbm2():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df, f3 = fit_lgbm_model(df, f2, target_col='sales', wgt_col='one')
    df['pred_sales'] = df['ypred']

    test_start = f3['test_start']
    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='lgbm2')
    return {'mae': mae, 'name': 'lgbm2'}


def main_lgbm3():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)
    df, f3 = fit_lgbm_model(df, f2, target_col='sales_log', wgt_col='one')
    df['pred_sales_log'] = df['ypred']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    test_start = f3['test_start']
    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='lgbm3')
    return {'mae': mae, 'name': 'lgbm3'}


def main_naive_1y_lag():
    df, f1 = get_data()
    df, f2 = add_features(df, f1)
    # we clip the YoY rate to 1 => 2.7
    df['sales_lag_vs_ewm_lag1y'] = df['sales_lag_vs_ewm_lag1y'].clip(lower=-1.0, upper=1.0)
    df['pred_sales_log'] = df['sales_lag_vs_ewm_lag1y']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    test_start = pd.to_datetime('2016-09-12')
    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1y')

    return {'mae': mae, 'name': 'naive_1y_lag'}


def main_naive_ewm():
    df, f1 = get_data()
    df, f2 = add_features(df, f1)
    df['pred_sales_log'] = df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    test_start = pd.to_datetime('2016-09-12')
    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naivecst')

    return {'mae': mae, 'name': 'naive_cst'}


def main_naive_ewm2():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['pred_sales'] = df['wgt']

    test_start = pd.to_datetime('2016-09-12')
    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naivecst')

    return {'mae': mae, 'name': 'naive_cst2'}


def main():
    r = []
    r += [main_lgbm1()]
    r += [main_lgbm2()]
    r += [main_lgbm3()]
    r += [main_naive_1y_lag()]
    r += [main_naive_ewm()]
    r += [main_naive_ewm2()]
    rdf = pd.DataFrame(r)
    print('-'*20)
    print('-'*20)
    print(rdf.sort_values('mae'))
    #            mae          name
    # 1  2831.593399  naive_1y_lag
    # 0  4555.135366         lgbm1
    # 2  4919.319504     naive_cst


# ipython -i -m crptmidfreq.season.kagstore_main
if __name__ == '__main__':
    main()
