import pandas as pd
import argparse
import random
import time
import os
import duckdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#import sys
#sys.path.append(os.path.abspath(os.getcwd()+'/..'))

from config_loc import get_data_db_folder
from featurelib.lib_v1 import *

g_folder = 'res_kmeans_v1'

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
    
# ipython -i  ./res/kmeans_manual.py
if __name__=='__main__':
    con = duckdb.connect(os.path.join(get_data_db_folder(),"my_database.db"),read_only=True)
    df=con.execute('''SELECT close_time,dscode,close,volume,taker_buy_volume 
                   FROM klines
                   WHERE close_time>'2024-03-01';
                   ''').df()

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
    clip_tret = 200e-4
    featd,nfeats=perform_clip(featd=featd,feats=['tret'],
                              high_clip=clip_tret,low_clip=-clip_tret,
                              folder=g_folder,name='None')

    ## adding the weight ewm(volume)
    featd,nfeats=perform_ewm(featd=featd,feats=['volume'],windows=[1000],folder=g_folder,name='None')
    featd['wgt'] = featd[nfeats[0]]

    ## pivotting for correlation matrix
    pdts,pfeatd  = perform_pivot(featd=featd,feats=['tret'],folder=g_folder,name='None')
    pX =  np.array([v for k,v in pfeatd.items()])
    pX = np.nan_to_num(pX)

    silhouette_method(pX, k_min=6, k_max=50)

    n_clusters = 10
    # For clustering, we can use the correlation values or the distance
    # In this example, we cluster on the correlation matrix rows.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init='auto')
    # Fit on the correlation matrix (using the rows as features)
    clusters = kmeans.fit_predict(pX)

    # Create a DataFrame with cluster labels and variable names
    cluster_df = pd.DataFrame({'dscode':  [k for k,v in pfeatd.items()], 'cluster': clusters})
    cluster_df['dscode_str']=cluster_df['dscode'].map(dscode_map)
    cluster_df = cluster_df.sort_values(by='cluster')
    cluster_df.to_csv('clusters.csv')