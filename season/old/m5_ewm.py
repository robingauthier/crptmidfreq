import os

import numpy as np
import pandas as pd

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.season.m5_data import evaluate_model, get_data
from crptmidfreq.utils.log import get_logger

log = get_logger()

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

    # This will be our target log(sales/lag(ewm(sales)))
    featd, ewmsales = perform_ewm(featd, feats=lagsales, windows=[20], **defargs)
    featd['target_sales_log_vs_ewm'] = featd[lagsales[0]]-featd[ewmsales[0]]
    featd['wgt'] = featd[ewmsales[0]]
    ndf = pd.DataFrame(featd)

    ndf['pred_sales_log'] = ndf['wgt']
    return ndf, []


# ipython -i -m crptmidfreq.season.m5_ewm
if __name__ == '__main__':
    df = get_data()
    df, feats = add_features(df)

    test_start = pd.to_datetime('2015-05-01')
    df_test = df.loc[lambda x:x['date'] >= test_start]
    evaluate_model(df_test, 'ewm_global')
    # MAE model ewm_global is mae:1.15  -- cnt:27230 -- date>=2015-05-01 00:00:00
