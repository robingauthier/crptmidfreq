import pandas as pd
import argparse
import random
import time
import os
import duckdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from crptmidfreq.utils.common import to_csv
from crptmidfreq.utils.log import get_logger


#import sys
#sys.path.append(os.path.abspath(os.getcwd()+'/..'))

from crptmidfreq.config_loc import get_data_db_folder
from crptmidfreq.featurelib.lib_v1 import *

g_folder = 'res_kmeans_v1'
logger = get_logger(__name__)  # Use module name for clarity

def silhouette_method(X, k_min=2, k_max=10):
    """
    Compute and plot the silhouette score for K-means over a range of cluster counts.
    A higher score means better-defined clusters.
    """
    print('Computing optimal K using silhouette score')
    ks = range(k_min, k_max + 1)
    silhouettes = []

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouettes.append(score)
    
    rdf=pd.DataFrame({'k':ks,'silhouette':silhouettes})
    rdf.to_csv('kmean_silhouette.csv')
    
def get_pivoted_data(tokens=['BTCUSDT','ETHUSDT'],sdate_str='2019-01-01'):
    logger.info('get_pivoted_data for tokens')
    clean_folder(g_folder)
    print('Reading data from DuckDB')
    con = duckdb.connect(os.path.join(get_data_db_folder(),"my_database.db"),read_only=True)
    list_tokens_str='\',\''.join(tokens)
    list_tokens_str='(\''+list_tokens_str+'\')'

    df=con.execute(f'''SELECT close_time,dscode,close,volume,taker_buy_volume 
                   FROM klines
                   WHERE 
                   close_time>'{sdate_str}' AND
                   dscode IN {list_tokens_str};
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
    #preclip_stats=(pd.Series(featd['tret']).describe()*1e4).astype(np.int64)
    clip_fast=True
    if clip_fast:
        clip_tret = 800e-4 # 8pct
        featd,nfeats=perform_clip(featd=featd,feats=['tret'],
                                high_clip=clip_tret,low_clip=-clip_tret,
                                folder=g_folder,name='None')
    else:
        featd,nfeats=perform_clip_quantile(featd, feats=['tret'],
                                low_clip=0.05,high_clip=0.95,
                                folder=g_folder,name='None')
    
    ## adding the weight ewm(volume)
    featd,nfeats=perform_ewm(featd=featd,feats=['volume'],windows=[1000],folder=g_folder,name='None')
    featd['wgt'] = featd[nfeats[0]]

    ## pivotting for correlation matrix
    pdts,pfeatd  = perform_pivot(featd=featd,feats=['tret'],folder=g_folder,name='None')
    pX =  np.array([v for k,v in pfeatd.items()])
    pX = np.nan_to_num(pX)
    
    pdft=pd.DataFrame(np.transpose(pX))
    pdft.columns=pdft.columns.map(dscode_map)
    pdft.index = pdts
    
    pdft.index=pd.to_datetime(pdft.index*1e3)
    logger.info('get_pivoted_data is ready')
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

# ipython -i  ./res/mr_cluster_v1/kmeans_manual.py
if __name__=='__main__':
    pX= get_pivoted_data()
    print(pX.head())