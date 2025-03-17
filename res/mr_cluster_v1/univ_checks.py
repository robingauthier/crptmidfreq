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
from config_loc import get_analysis_folder
from featurelib.lib_v1 import *

g_folder = 'res_kmeans_v1'


def get_univ_cnt(sdate_str='2019-01-01'):
    print('Reading data from DuckDB')
    con = duckdb.connect(os.path.join(get_data_db_folder(),"my_database.db"),read_only=True)

    df = con.execute(f'''
        SELECT CAST(close_time AS DATE) AS trade_date, COUNT(DISTINCT dscode) AS unique_dscode_count
        FROM klines
        WHERE close_time > '{sdate_str}' 
        GROUP BY trade_date;
        ''').df()
    tpath=os.path.join(get_analysis_folder(),'univ_cnt.csv')
    print(f'Please open :{tpath}')
    df.to_csv(tpath)
    return df

# ipython -i  ./res/mr_cluster_v1/univ_checks.py
if __name__=='__main__':
    pX= get_univ_cnt()
