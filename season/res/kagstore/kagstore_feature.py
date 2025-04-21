import os

import numpy as np
import pandas as pd

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.season.res.kagstore.kagstore_data import get_data
from crptmidfreq.season.yoy import deseasonalize_yoy
from crptmidfreq.season.yoy_hol import deseasonalize_yoy_hol
from crptmidfreq.utils.common import pandas_to_dict, to_csv
from crptmidfreq.utils.log import get_logger

log = get_logger()
g_folder = os.path.join(get_feature_folder(), 'kagstore')+'/'


def add_features(df, f1, use_log=True, use_hols=False, use_norm=False):

    assert 'dtsi' in df.columns
    assert 'dscode' in df.columns

    df['dscode'] = df['dscode'].astype(np.int64)
    df['dtsi'] = df['dtsi'].astype(np.int64)

    if use_norm:
        mdf = df.groupby('dscode')[['sales']].mean().reset_index()\
            .rename(columns={'sales': 'mean_sales'})
        df = df.merge(mdf, on='dscode', how='left')
        df['sales'] = np.where(df['mean_sales'] > 1, df['sales']/df['mean_sales'], 1.0)

    defargs = {'folder': g_folder}
    featd = pandas_to_dict(df)

    # lagging the target
    f0 = ['sales_log'] if use_log else ['sales']
    featd, lagsales = perform_lag(featd, feats=f0, windows=[1], **defargs)

    # This will be our target log(sales/lag(ewm(sales)))
    featd, ewmsales = perform_ewm(featd, feats=lagsales, windows=[10], **defargs)
    featd['sales_vs_ewm'] = featd[f0[0]]-featd[ewmsales[0]]
    featd['wgt'] = featd[ewmsales[0]]
    featd['one'] = np.ones(featd['dtsi'].shape[0])

    # Feature 1 : sales/ewm(sales)
    featd['sales_lag_vs_ewm'] = featd[lagsales[0]]-featd[ewmsales[0]]

    ndf = pd.DataFrame(featd)
    # Feature 4 : 1 year lag on same day

    func = deseasonalize_yoy_hol if use_hols else deseasonalize_yoy
    ndf, salyoy = func(ndf,
                       date_col='date',
                       stock_col='dscode',
                       serie_col='sales_vs_ewm',  # no need to use the lagged version
                       operation='lag')
    ndf, salyoy2 = func(ndf,
                        date_col='date',
                        stock_col='dscode',
                        serie_col=f0[0],  # no need to use the lagged version
                        operation='lag')
    ndf, salyoy3 = func(ndf,
                        date_col='date',
                        stock_col='dscode',
                        serie_col=f0[0],  # no need to use the lagged version
                        operation='diff')
    ndf, salyoy4 = func(ndf,
                        date_col='date',
                        stock_col='dscode',
                        serie_col=lagsales[0],  # no need to use the lagged version
                        operation='diff')
    nfeats = lagsales+salyoy+salyoy2+['sales_lag_vs_ewm']+['wgt']+salyoy4

    f2 = {
        'categorical': f1['categorical'],
        'numerical': nfeats,
        'agglevel': f1['agglevel'],
    }

    return ndf, f2


def add_features_lag(df, f1, use_log=True, use_hols=False, n_lags=20):
    assert 'dtsi' in df.columns
    assert 'dscode' in df.columns

    df['dscode'] = df['dscode'].astype(np.int64)
    df['dtsi'] = df['dtsi'].astype(np.int64)
    df['sales'] = df['sales'].replace(0.0, np.nan)
    defargs = {'folder': g_folder}
    featd = pandas_to_dict(df)
    list_lags = range(1, n_lags)

    featd, ff = perform_ffill(featd, feats=['sales'], **defargs)
    featd = rename_key(featd, ff[0], 'sales')

    # lagging the target
    f0 = ['sales_log'] if use_log else ['sales']
    featd, lagsales = perform_lag(featd, feats=f0, windows=list_lags, **defargs)
    ndf = pd.DataFrame(featd)

    f2 = {
        'categorical': [],
        'numerical': lagsales,
        'agglevel': [],
    }

    return ndf, f2


# ipython -i -m crptmidfreq.season.kagstore_feature
if __name__ == '__main__':
    df, f1 = get_data(agglevel=None)
    df, f2 = add_features(df, f1)
    #to_csv(df.loc[lambda x:x['family'] == 'AUTOMOTIVE'], 'example_features_kagstore_AUTOMOTIVE')
    to_csv(df.loc[lambda x:x['family'] == 'FROZEN FOODS'], 'example_features_kagstore_FROZEN_FOODS')
