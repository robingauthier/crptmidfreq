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


def get_data(agglevel='dept_id+store_id'):
    """
    dept_id
    FOODS_1        2160
    FOODS_2        3980
    FOODS_3        8230
    HOBBIES_1      4160
    HOBBIES_2      1490
    HOUSEHOLD_1    5320
    HOUSEHOLD_2    5150

    """
    log.info('season - get data m5 competition')

    raw_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data/'))

    train_df = pd.read_csv(raw_data_dir+'/M5-sales_train_evaluation.csv')  # , nrows=1_000
    prices_df = pd.read_csv(raw_data_dir+'/M5-sell_prices.csv')
    calendar_df = pd.read_csv(raw_data_dir+'/M5-calendar.csv')

    index_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    grid_df = pd.melt(train_df,
                      id_vars=index_columns,
                      var_name='d',
                      value_name='sales')
    if agglevel == 'dept_id+store_id':
        grid_df = grid_df\
            .groupby(['dept_id', 'cat_id', 'store_id', 'state_id', 'd'])\
            .agg({'sales': 'sum'})\
            .reset_index()
    grid_df['sales_log'] = np.log(grid_df['sales']+1)

    calendar_df = calendar_df.reset_index().rename(columns={'index': 'd'})
    calendar_df['dtsi'] = calendar_df['d'].astype(np.int64)
    calendar_df['d'] = 'd_'+calendar_df['d'].astype(str)
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    cal_cols = ['d', 'date', 'dtsi', 'wm_yr_wk', 'weekday', 'wday', 'month']
    grid_df = grid_df.merge(calendar_df[cal_cols], on='d', how='left')

    # adding calendar features
    cal_df = event_distances(start_date=pd.to_datetime('2010-01-01'),
                             end_date=pd.to_datetime('2027-01-01'))
    grid_df = grid_df.merge(cal_df, on='date', how='left')

    prices_df['dept_id'] = prices_df['item_id'].str[:-4]
    if agglevel == 'dept_id+store_id':
        prices_df = prices_df\
            .groupby(['dept_id', 'wm_yr_wk', 'store_id'])\
            .agg({'sell_price': 'mean'})\
            .reset_index()

    # adding prices
    grid_df = grid_df.merge(prices_df, on=['dept_id', 'wm_yr_wk', 'store_id'], how='left')
    grid_df['sell_price_log'] = np.log(grid_df['sell_price']+1)

    # sorting now
    grid_df = grid_df.sort_values(['date'])
    return grid_df


def pandas_to_dict(df):
    r = {}
    for col in df.columns:
        r[col] = df[col].values
    return r


def dump_solution_loc(df, dept='HOUSEHOLD_1', store='WI_2'):
    tdf = df\
        .loc[lambda x:x['dept_id'] == dept]\
        .loc[lambda x:x['store_id'] == store]\
        .loc[:, ['date', 'pred_sales_log', 'sales_log']]
    tdf = tdf.sort_values('date')
    tdf['diff'] = tdf['sales_log']-tdf['pred_sales_log']
    tdf['cum_diff'] = tdf['diff'].cumsum()
    to_csv(tdf, f'example_{dept}_{store}')


def dump_solution(df):
    dump_solution_loc(df, dept='HOUSEHOLD_1', store='WI_2')
    dump_solution_loc(df, dept='HOBBIES_1', store='WI_2')
    dump_solution_loc(df, dept='FOODS_2', store='WI_2')
    dump_solution_loc(df, dept='FOODS_3', store='WI_2')
    dump_solution_loc(df, dept='HOUSEHOLD_2', store='WI_2')


def evaluate_model(df, name=''):
    first_date = df['date'].min()

    mae = (df['wgt']*(df['sales_log']-df['pred_sales_log']).abs()).mean()
    nbpoints = df.shape[0]
    print(f'MAE model {name} is mae:{mae:.2f}  -- cnt:{nbpoints} -- date>={first_date}')
    return mae
