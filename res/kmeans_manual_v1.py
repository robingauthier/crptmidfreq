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

def get_pivoted_data(start_date='2025-03-01',end_date='2026-01-01'):
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
    featd,nfeats=perform_ewm(featd=featd,feats=['volume'],windows=[1000],folder=g_folder,name='None')
    featd['wgt'] = featd[nfeats[0]]

    ## rank cross sectionally by volume to build a robust universe
    perform_cs_rank()

    ## adding the excess volume
    featd['excess_volume'] = np.divide(
                    featd[nfeats[0]],
                    featd['volume'],
                    out=np.zeros_like(featd['volume']),
                    where=~np.isclose(featd['volume'], np.zeros_like(featd['volume'])))
    perform_clip(featd=featd)
    

    ## pivotting for correlation matrix / clustering
    pdts,pfeatd  = perform_pivot(featd=featd,feats=['tret'],folder=g_folder,name='None')
    pX =  np.array([v for k,v in pfeatd.items()])
    pX = np.nan_to_num(pX)
    
    pdft=pd.DataFrame(np.transpose(pX))
    pdft.columns=pdft.columns.map(dscode_map)
    pdft.index = pdts
    return pdft

def main():
    pX= get_pivoted_data()

    #silhouette_method(pX, k_min=6, k_max=50)

    n_clusters = 30
    # For clustering, we can use the correlation values or the distance
    # In this example, we cluster on the correlation matrix rows.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init='auto')
    # Fit on the correlation matrix (using the rows as features)
    clusters = kmeans.fit_predict(pX.T)

    # Create a DataFrame with cluster labels and variable names
    cluster_df = pd.DataFrame({'dscode_str':  pX.columns.tolist(), 'cluster': clusters})
    cluster_df = cluster_df.sort_values(by='cluster')
    to_csv(cluster_df,'clusters')

### faudrait liquidite, start-trading,end-trading dans la data !

# ipython -i  ./res/kmeans_manual.py
if __name__=='__main__':
    main()
    
    # TODO:bucketplot of volume traded / market cap vs P&L yep
    # perform a rolling kmeans
    # 