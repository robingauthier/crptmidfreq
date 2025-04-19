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

log = get_logger()


def get_data(agglevel='family'):
    """
    (Pdb) p train_df['family'].value_counts()
    family
    AUTOMOTIVE                    90936
    HOME APPLIANCES               90936
    SCHOOL AND OFFICE SUPPLIES    90936
    PRODUCE                       90936
    PREPARED FOODS                90936
    ...
    """
    log.info('season - get data kaggle retail')

    raw_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data/'))

    df = pd.read_csv(raw_data_dir+'/kag-store-sales-train.csv')
    if agglevel == 'family':
        df = df.groupby(['family', 'date']).agg({'sales': 'sum', 'onpromotion': 'sum'}).reset_index()
        df['store_nbr'] = 1
    df['date'] = pd.to_datetime(df['date'])
    df['dtsi'] = df['date'].astype(np.int64)/1e9/3600/24
    df['wday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weeknum'] = df['date'].dt.isocalendar().week

    df['sales_log'] = np.log(df['sales']+1)

    # adding calendar features
    cal_df = event_distances(start_date=pd.to_datetime('2010-01-01'),
                             end_date=pd.to_datetime('2027-01-01'))
    df = df.merge(cal_df, on='date', how='left')

    df['dscode_str'] = df['family']+'_'+df['store_nbr'].astype(str)
    df['dscode'] = pd.Categorical(df['dscode_str']).codes

    # sorting now
    df = df.sort_values(['date'])

    featnames = {
        'categorical': ['family',
                        'store_nbr',
                        'wday', 'month', 'day', 'weeknum',
                        'evt_all_next', 'evt_all_prev',
                        'dscode',
                        ],
        'numerical': ['sales_log', 'sales', 'onpromotion',
                      'dist_since_last_all',
                      'dist_to_next_all',
                      'dist_since_last_High',
                      'dist_to_next_High'
                      ],
        'agglevel': agglevel,
    }

    return df, featnames


def evaluate_model(df, name=''):
    first_date = df['date'].min()
    #mae = (df['wgt']*(df['sales']-df['pred_sales']).abs()).mean()
    mae = (df['sales']-df['pred_sales']).abs().mean()
    nbpoints = df.shape[0]
    print(f'MAE model {name} is mae:{mae:.2f}  -- cnt:{nbpoints} -- date>={first_date}')
    return mae


# ipython -i -m crptmidfreq.season.kagstore_data
if __name__ == '__main__':
    df, f = get_data()
    to_csv(df.loc[lambda x:x['family'] == 'LIQUOR,WINE,BEER'], 'example')
