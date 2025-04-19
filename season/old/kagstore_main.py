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
from mllib.lgbm_sklearn import fit_lgbm_model
from crptmidfreq.season.gluonts_v1 import fit_gluonts_model

# sales_log= log(1+sales)
# exp(sales_log)= 1+sales
# sales= exp(sales_log)-1

test_start = pd.to_datetime('2016-09-01')


def dump_data(df, name=''):
    wcols = ['date', 'sales', 'pred_sales']
    fams = ['LIQUOR,WINE,BEER', 'AUTOMOTIVE', 'FROZEN FOODS']
    for fam in fams:
        tdf = df.loc[lambda x:x['family'] == fam][wcols].sort_values('date')
        tdf['d'] = tdf['sales']-tdf['pred_sales']
        tdf['cd'] = tdf['d'].cumsum()
        to_csv(tdf, f'example_kagstore_{name}_{fam}')


def main_gluonts_ff():
    from gluonts.mx import SimpleFeedForwardEstimator, Trainer
    # /Users/sachadrevet/anaconda3/lib/python3.11/site-packages/gluonts/mx/model/simple_feedforward/_network.py

    # first dense layer will be context_length x num_hidden_dimensions[0]
    # second dense layer will be num_hidden_dimensions[0] x num_hidden_dimensions[1]
    # last layer will be num_hidden_dimensions[-1] x prediction_length

    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10, 10],  # this means 2 hidden layers
        prediction_length=1,  # forecast_its[ii].samples.shape[1]
        context_length=20,  # forecast_its[ii].samples.shape[0]
        trainer=Trainer(ctx="cpu",
                        epochs=10,
                        learning_rate=1e-2,
                        weight_decay=1e-4,
                        num_batches_per_epoch=100),
    )

    df, f1 = get_data()
    df, f3 = fit_gluonts_model(df,
                               f1,
                               estimator,
                               target_col='sales',
                               name='gluonts_ff',
                               )
    df['pred_sales'] = df['gluonts_ff_predh0_mean']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='gluonts_ff')
    return {'mae': mae, 'name': 'gluonts_ff'}


def main_gluonts_nbeats():
    from gluonts.mx import NBEATSEstimator
    from gluonts.mx import Trainer

    estimator = NBEATSEstimator(
        num_stacks=2,
        widths=[5],  # len is 1 or num_stacks
        num_blocks=[1],  # len is 1 or num_stacks
        prediction_length=1,
        context_length=10,
        freq="D",
        trainer=Trainer(ctx="cpu",
                        epochs=20,
                        learning_rate=1e-2,  # with 1e-1 we get divergence issues
                        weight_decay=1e-4,
                        num_batches_per_epoch=100),
    )

    df, f1 = get_data()

    df, f3 = fit_gluonts_model(df,
                               f1,
                               estimator,
                               target_col='sales',
                               name='gluonts_nbeats',
                               )
    df['pred_sales'] = df['gluonts_nbeats_predh0_mean']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='gluonts_nbeats')
    return {'mae': mae, 'name': 'gluonts_nbeats'}


def main_lgbm1():
    df, f1 = get_data()
    df, f2 = add_features(df, f1)
    df, f3 = fit_lgbm_model(df, f2, target_col='target_sales_vs_ewm', wgt_col='wgt')
    df['pred_sales_log'] = df['ypred']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='lgbm1')
    return {'mae': mae, 'name': 'lgbm1'}


def main_lgbm2():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df, f3 = fit_lgbm_model(df, f2, target_col='sales', wgt_col='one')
    df['pred_sales'] = df['ypred']

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

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1y')

    return {'mae': mae, 'name': 'naive_1y_lag'}


def main_naive_ewm():
    df, f1 = get_data()
    df, f2 = add_features(df, f1)
    df['pred_sales_log'] = df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naivecst')

    return {'mae': mae, 'name': 'naive_cst'}


def main_naive_ewm2():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['pred_sales'] = df['wgt']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naivecst')

    return {'mae': mae, 'name': 'naive_cst2'}


def main():
    """
               mae          name
2   404.441263         lgbm2
3  1067.753716         lgbm3
4  2799.126016  naive_1y_lag
0  2888.637159    gluonts_ff
1  4590.787652         lgbm1
5  4922.202226     naive_cst
6  5058.685658    naive_cst2
"""
    r = []
    r += [main_gluonts_nbeats()]
    r += [main_gluonts_ff()]
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
    # main_gluonts_ff()
    # main_gluonts_nbeats()
