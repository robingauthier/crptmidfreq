import pandas as pd
import argparse
import random
import time
import os
import duckdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from crptmidfreq.utils.common import to_csv
from crptmidfreq.utils.common import rename_key
from numba.typed import Dict
from numba.core import types

from crptmidfreq.config_loc import get_data_db_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import get_sig_cols
from crptmidfreq.utils.common import ewm_alpha

g_folder = 'res_kmeans_v1'
logger = get_logger()

def main(start_date='2025-03-01',end_date='2026-01-01'):
    logger.info(f'mr_cluster start_date={start_date} end_date={end_date}')
    # all the hyper parameters
    window_volume_wgt = 60*24*30
    window_volume_univ = 60*24*20
    window_volume_excess_slow = 60*20*10
    window_volume_excess_fast = 60*20
    windows_sret_ewm = [100,200,800,1000]
    windows_sharpe = [2000]
    volume_excess_clip_high=5.0
    volume_excess_clip_low=0.1
    kmeans_lookback=24*60*30
    kmeans_fitfreq=24*60*10
    universe_count = 100
    ipo_burn = 60*24
    
    r=StepperRegistry()

    logger.info('Reading data from DuckDB')
    con = duckdb.connect(os.path.join(get_data_db_folder(),"my_database.db"),read_only=True)
    df=con.execute(f'''SELECT close_time,dscode,close,volume,taker_buy_volume 
                   FROM klines
                   WHERE CAST(close_time AS DATE)>='{start_date}'
                   AND CAST(close_time AS DATE)<='{end_date}';
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
    
    # Casing to float64 -- otherwise we have issues later
    featd,_=perform_cast_float64(featd, feats=['close','volume','taker_buy_volume'], folder=g_folder,r=r)

    # turnover = volume*close  should be in USDT
    featd['turnover']=featd['volume']*featd['close']
    
    ## adding distance since IPO  :: cnt_exists
    featd,nfeats=perform_cnt_exists(featd=featd,feats=[],folder=g_folder,name='None')

    ## adding returns
    featd,nfeats=perform_diff(featd=featd,feats=['close'],windows=[1],folder=g_folder,name='None')
    featd['tret']=np.divide(
                    featd[nfeats[0]],
                    featd['close'],
                    out=np.zeros_like(featd['close']),
                    where=~np.isclose(featd['close'], np.zeros_like(featd['close'])))

    # Clip tret
    featd,nfeats=perform_clip_quantile_global(featd, feats=['tret'],
                                low_clip=0.05,high_clip=0.95,
                                folder=g_folder,name='None')
        
    ## adding the weight ewm(volume)
    featd,nfeats_ev=perform_ewm(featd=featd,feats=['turnover'],windows=[window_volume_wgt],folder=g_folder,name='None')
    featd=rename_key(featd,nfeats_ev[0],'wgt')

    # Removing the market in tret => tret_xmkt
    featd,nfeats=perform_cs_demean(featd=featd,feats=['tret'],by=None,wgt='wgt',folder=g_folder,name='None')
    featd,nfeats=perform_clip_quantile_global(featd=featd,feats=nfeats,
                              low_clip=0.05,high_clip=0.95,
                              folder=g_folder,name='None')
    featd['tret_xmkt'] = featd[nfeats[0]]
    
    ## rank cross sectionally by volume to build a robust universe
    featd,nfeats_ev=perform_ewm(featd=featd,feats=['turnover'],windows=[window_volume_univ],folder=g_folder,name='None')
    featd=rename_key(featd,nfeats_ev[0],'turnover_ewm')
    featd,nfeats=perform_cs_rank_int_decreasing(featd=featd,feats=['turnover_ewm'],folder=g_folder,name='None')

    ## define universe
    featd['univ']=1*(featd[nfeats[0]]<=universe_count)

    ## adding the excess turnover which is 1+x
    featd,nfeats_ev=perform_ewm(featd=featd,feats=['turnover'],
                                windows=[window_volume_excess_fast,window_volume_excess_slow],
                                folder=g_folder,name='None')
    featd,nfeats_d = perform_divide(featd,
                                    numcols=[nfeats_ev[0]],
                                    denumcols=[nfeats_ev[1]],
                                    folder=g_folder,name='None')
    featd,nfeats_d2= perform_clip(featd=featd,feats=nfeats_d,
                                  low_clip=volume_excess_clip_low,
                                  high_clip=volume_excess_clip_high,
                                  folder=g_folder,name='None')
    featd=rename_key(featd,nfeats_d2[0],'turnover_excess')
    
    ## taker_buy_turnover/turnover
    featd,nfeats=perform_divide(featd,['taker_buy_volume'],['volume'],folder=g_folder,name='None')
    for col in nfeats:
        featd[col]=featd[col]-0.5 # needs to be centered on 0.0
    featd,nfeats_tv=perform_ewm(featd=featd,feats=nfeats,windows=windows_sret_ewm,folder=g_folder,name='None')
    
    ## Bumping return by the excess turnover
    featd['tret_xturnover']=featd['turnover_excess']*featd['tret']
    featd['tret_xmkt_xturnover']=featd['turnover_excess']*featd['tret_xmkt']

    ## pivotting for correlation matrix / clustering
    pdts,pfeatd  = perform_pivot(featd=featd,feats=['tret_xmkt'],folder=g_folder,name='None')
    pX =  np.array([v for k,v in pfeatd.items()])
    pX = np.nan_to_num(pX)
    pX = np.transpose(pX) # ndts x ndscode
    pdscodes = list(pfeatd.keys())
    
    # Fitting the model now / directly calling stepper
    def model_gen_kmeans():
        return KMeans(n_clusters=20, random_state=42,n_init='auto')
    cls_model = KmeansStepper \
    .load(folder=g_folder, name="model_kmeans", 
            lookback=kmeans_lookback,
            minlookback=500,
            fitfreq=kmeans_fitfreq,
            gap=1,
            model_gen=model_gen_kmeans,
            with_fit=True)
    # kmeanres is 2dim array ndts x ndscode
    kmeansres=cls_model.update(pdts, pX, yserie=None,wgtserie=None)
    
    kmeansd = Dict.empty(
        key_type=types.int64,    # Define the type of keys
        value_type=types.Array(types.float64, 1, "C")  # Define the type of values
    )
    for i in range(len(pdscodes)):
        kmeansd[pdscodes[i]]=kmeansres[:,i].copy(order='C')
    
    # res is a matrix ndst x ndscode
    ndt,ndscode,nserie  = perform_unpivot(pdts,kmeansd,folder=g_folder,name='None')
    featd2 = {'dtsi':ndt,'dscode':ndscode,'kmeans_cat':nserie}
    
    
    featd,nfeats=perform_merge_asof(featd, featd2, feats=['kmeans_cat'], folder=g_folder,name='None')
    featd=rename_key(featd,nfeats[0],'kmeans_cat')
    
    # Now removing the cluster mean
    featd['wgt_xuniv'] = featd['wgt']*featd['univ']
    featd,nfeats=perform_cs_demean(featd=featd,feats=['tret_xmkt'],by='kmeans_cat',
                                   wgt='wgt_xuniv',
                                   folder=g_folder,name='None')
    featd=rename_key(featd,nfeats[0],'sret_kmeans')
    
    # computing different ewms of the residuals
    featd,nfeats_ewm_sr=perform_ewm(featd=featd,feats=['sret_kmeans'],
                                    windows=windows_sret_ewm,
                                    folder=g_folder,name='None')
    
    # computing the volatility of the sret_kmeans
    featd,nfeats_vol_sr=perform_ewm_std(featd=featd,feats=['sret_kmeans'],
                                    windows=windows_sret_ewm,
                                    folder=g_folder,name='None')
    featd,_=perform_avg_features_fillna0(featd,xcols=nfeats_vol_sr,outname='mual_std',folder=g_folder,name='None')
    nfeats_zs=[]
    for i in range(len(windows_sret_ewm)):
        ewm_col=nfeats_ewm_sr[i]
        win = windows_sret_ewm[i]
        alpha = 1-ewm_alpha(win)
        
        # ewm(X) has Var = Var(x_i)* (1-alpha)/(1+alpha)
        featd['todel_num']=featd[ewm_col]*np.sqrt((1+alpha)/(1-alpha))
        featd,nfeats = perform_divide(featd,['todel_num'],['mual_std'],folder=g_folder,name='None')
        featd,nfeats = perform_clip(featd=featd,
                                   feats=[nfeats[0]],
                                   low_clip=-3.0,high_clip=3.0,
                                   folder=g_folder,name='None')
        featd=rename_key(featd,nfeats[0],f'zs_{win}')
        nfeats_zs+=[f'zs_{win}']

    # Here we create the column mual
    featd,_=perform_avg_features_fillna0(featd,xcols=nfeats_zs,outname='mual_temp',folder=g_folder,name='None')
    featd,nfeats =perform_clip(featd=featd,feats=['mual_temp'],low_clip=-3.0,high_clip=3.0,folder=g_folder,name='None')
    featd=rename_key(featd,nfeats[0],'mual')
    
    # Changing the sign now for mual
    featd['mual']=-featd['mual']
    
    # computing the sharpe of the mual
    featd,nfeats=perform_lag(featd,feats=['mual'],windows=[1],folder=g_folder,r=r)
    featd[f'mual_pnl']=featd[nfeats[0]]*featd['tret_xmkt']
    featd,nfeats_num_sharpe = perform_ewm(featd,['mual_pnl'],windows=windows_sharpe,folder=g_folder,r=r)
    featd,nfeats_denum_sharpe = perform_ewm_std(featd,['mual_pnl'],windows=windows_sharpe,folder=g_folder,r=r)
    featd,nfeats_sharpe=perform_divide(featd,numcols=nfeats_num_sharpe,denumcols=nfeats_denum_sharpe,folder=g_folder,r=r)
    
    # Computing a quantile on sharpe
    featd,nfeats_sharpe_qtl = perform_quantile_global(featd, feats=nfeats_sharpe, qs=[0.4],folder=g_folder,r=r)
    featd['mual_high_sharpe']=featd['mual']*(featd[nfeats_sharpe[0]]>featd[nfeats_sharpe_qtl[0]])
    
    # Creating the signal
    zs_cols=[x for x in featd.keys() if x.startswith('zs_')]
    featd,_ = perform_to_sig(featd,feats=zs_cols,folder=g_folder,r=r)
    featd = rename_key(featd,'mual','sig_mual')
    featd = rename_key(featd,'mual_high_sharpe','sig_mual_high_sharpe')
    
    # removing post IPO   20days
    for col in get_sig_cols(featd):
        featd[col] = featd[col]*(featd['cnt_exists']>ipo_burn)
    
    ## adding the forward return
    featd,nfeats = perform_lag_forward(featd=featd,feats=['tret_xmkt'],windows=[-1])
    featd=rename_key(featd,nfeats[0],'forward_fh1')
    
    
    return featd

def filter_dict_to_dscode(featd,dscode_str='BTCUSDT'):
    """filter the dictionary to a specific dscode"""
    filter_numpy = (featd['dscode_str']==dscode_str)
    featd2 = {k:featd[k][filter_numpy] for k in featd.keys()}
    return featd2

def filter_dict_to_dts(featd,dtsi=1):
    """filter the dictionary to a specific dscode"""
    filter_numpy = (featd['dtsi']==dtsi)
    featd2 = {k:featd[k][filter_numpy] for k in featd.keys()}
    return featd2

def dump_extract(featd):
    """Dumping the data for manual checks"""
    logger.info('Dumping the data for manual checks')
    
    icols=['dtsi','dscode_str','close','sig_mual','univ','kmeans_cat','turnover_excess']
    df=pd.DataFrame({k:v for k,v in featd.items() if k in icols})
    df['dtsi']=pd.to_datetime(df['dtsi']*1e3)
    
    df_cs_loc=df[df['dtsi']==df['dtsi'].max()]
    to_csv(df_cs_loc,'df_cs_loc')
    
    df_ts_loc = df[df['dscode_str']=='BCHUSDT']
    to_csv(df_ts_loc,'df_ts_loc')
    
    df_g=df.assign(cnt=1)\
        .groupby('dtsi')\
        .agg({'sig_mual':('mean','std'),'univ':'sum','cnt':'sum'})
    to_csv(df_g,'df_univ_cnt')
    
    # all the columns now
    featd_loc = filter_dict_to_dscode(featd,dscode_str='BCHUSDT')
    df_loc=pd.DataFrame(featd_loc)
    df_loc['dtsi']=pd.to_datetime(df_loc['dtsi']*1e3)
    to_csv(df_loc,'df_ts_loc_allcols')
    

    dtsi=np.max(featd['dtsi'])
    featd_loc = filter_dict_to_dts(featd,dtsi=dtsi)
    df_loc=pd.DataFrame(featd_loc)
    df_loc['dtsi']=pd.to_datetime(df_loc['dtsi']*1e3)
    to_csv(df_loc,'df_cs_loc_allcols')


def bktest(featd):
    # TODO:bucketplot of turnover traded / market cap vs P&L yep
    # perform a rolling kmeans
    stats=perform_bktest(featd,folder=g_folder,name="None")
    #import pdb;pdb.set_trace()
    #perform_bucketplot()
    
# ipython -i -m crptmidfreq.res.mr_cluster_v1.kmeans_manual_v1
if __name__=='__main__':
    clean_folder(g_folder)
    featd=main(start_date='2025-02-01',end_date='2026-01-01')
    bktest(featd=featd)
    dump_extract(featd)