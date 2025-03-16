import pandas as pd
import argparse
import random
import time
import os
import duckdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.common import to_csv

#import sys
#sys.path.append(os.path.abspath(os.getcwd()+'/..'))

from config_loc import get_data_db_folder
from featurelib.lib_v1 import *

g_folder = 'res_kmeans_v1'

def main(start_date='2025-03-01',end_date='2026-01-01'):
    clean_folder(g_folder)
    print('Reading data from DuckDB')
    con = duckdb.connect(os.path.join(get_data_db_folder(),"my_database.db"),read_only=True)
    df=con.execute('''SELECT close_time,dscode,close,volume,taker_buy_volume 
                   FROM klines
                   WHERE close_time>'2024-03-01';
                   ''').df()

    print('Cleaning data')
    # we need to convert dscode to an integer 
    df['dscode_str']=df['dscode'].copy()
    cat =  pd.Categorical(df['dscode_str'])
    df['dscode']=cat.codes
    df['dscode'] = df['dscode'].astype('int64') # defaults to int16
    dscode_map = dict(enumerate(cat.categories))
    df['dtsi'] = df['close_time'].astype('int64')
 
    df.sort_values('close_time',ascending=True,inplace=True)

    # working on numpy now
    featd={col: df[col].values for col in df.columns}

    ## adding returns
    featd,nfeats=perform_diff(featd=featd,feats=['close'],windows=[1],folder=g_folder,name='None')
    featd['tret']=np.divide(
                    featd[nfeats[0]],
                    featd['close'],
                    out=np.zeros_like(featd['close']),
                    where=~np.isclose(featd['close'], np.zeros_like(featd['close'])))

    # Clip tret
    featd,nfeats=perform_clip_quantile(featd, feats=['tret'],
                                low_clip=0.05,high_clip=0.95,
                                folder=g_folder,name='None')
        
    ## adding the weight ewm(volume)
    window_1month = 60*24*30
    featd,nfeats=perform_ewm(featd=featd,feats=['volume'],windows=[window_1month],folder=g_folder,name='None')
    featd['wgt'] = featd[nfeats[0]]

    # Removing the market 
    featd,nfeats=perform_cs_demean(featd=featd,feats=['volume'],by=None,wgt='wgt',folder=g_folder,name='None')
    featd['tret_xmkt'] = featd[nfeats[0]]
    
    ## rank cross sectionally by volume to build a robust universe
    featd,nfeats=perform_cs_rank(featd=featd,feats=['wgt'],folder=g_folder,name='None')

    ## define universe
    featd['univ']=1*(featd[nfeats[0]]<=50)


    ## adding the excess volume
    featd['excess_volume'] = np.divide(
                    featd[nfeats[0]],
                    featd['volume'],
                    out=np.zeros_like(featd['volume']),
                    where=~np.isclose(featd['volume'], np.zeros_like(featd['volume'])))
    perform_clip(featd=featd)
    
    ## Bumping return by the excess volume
    featd['tret_volume']=featd['excess_volume']*featd['tret']

    ## pivotting for correlation matrix / clustering
    pdts,pfeatd  = perform_pivot(featd=featd,feats=['tret'],folder=g_folder,name='None')
    pX =  np.array([v for k,v in pfeatd.items()])
    pX = np.nan_to_num(pX)
    
    def model_gen_kmeans():
        return KMeans(n_clusters=20, random_state=42,n_init='auto')
    
    cls_model = ModelStepper \
    .load(folder=f"{folder}", name=f"{name}_{xcols}_{wgt}_{ycol}", 
            lookback=lookback,minlookback=minlookback,
                fitfreq=fitfreq,gap=gap,
                model_gen=model_gen,with_fit=with_fit)
    res=cls_model.update(featd['dtsi'], xseries, yserie=yserie,wgtserie=wgtserie)

    
    featd,nfeats=perform_model(featd, feats=[], wgt=None,ycol=None,folder=None, name=None,
                  lookback=300,
                  minlookback=100,
                  fitfreq=10,
                  gap=1,
                  model_gen=model_gen_kmeans,
                  with_fit=True)
    
    pdft=pd.DataFrame(np.transpose(pX))
    pdft.columns=pdft.columns.map(dscode_map)
    pdft.index = pdts
    
    
    
    # TODO:bucketplot of volume traded / market cap vs P&L yep
    # perform a rolling kmeans
    perform_bucketplot()
    return pdft

# ipython -i  ./res/kmeans_manual.py
if __name__=='__main__':
    main()
