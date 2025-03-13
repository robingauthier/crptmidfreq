import pandas as pd
import argparse
import random
import time
import os
import duckdb

import sys
sys.path.append(os.path.abspath(os.getcwd()+'/..'))

from config_loc import get_data_folder
from featurelib.lib_v1 import *

con = duckdb.connect(os.path.join(get_data_folder(),"my_database.db"),read_only=True)

g_folder = 'res_exploration_v1'

df=con.execute('SELECT close_time,dscode,close,volume,taker_buy_volume FROM klines;').df()


# we need to convert dscode to an integer 
df['dscode_str']=df['dscode'].copy()
df['dscode']=pd.Categorical(df['dscode_str']).codes
df['dtsi'] = df['close_time'].astype('int64')

df.sort_values('close_time',ascending=True,inplace=True)

featd={col: df[col].values for col in df.columns}


## adding returns
featd,nfeats=perform_diff(featd=featd,feats=['close'],windows=[1],folder=g_folder,name='None')
featd['tret']=np.divide(
                featd[nfeats[0]],
                featd['close'],
                out=np.zeros_like(featd['close']),
                where=~np.isclose(featd['close'], np.zeros_like(featd['close'])))


## adding the weight ewm(volume)
featd,nfeats=perform_ewm(featd=featd,feats=['volume'],windows=[1000],folder=g_folder,name='None')
featd['wgt'] = featd[nfeats[0]]


