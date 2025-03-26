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
from functools import partial
from crptmidfreq.config_loc import get_data_db_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.strats import *
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import get_sig_cols, get_sigf_cols, get_forward_cols
from crptmidfreq.utils.common import ewm_alpha
from crptmidfreq.utils.common import filter_dict_to_univ
from crptmidfreq.utils.common import filter_dict_to_dscode
from crptmidfreq.utils.common import filter_dict_to_dts
from crptmidfreq.utils.common import save_features
from crptmidfreq.utils.common import save_signal


logger = get_logger()

g_folder = 'res_kmeans_v1'
g_r = StepperRegistry()


def main_features(start_date='2025-03-01', end_date='2026-01-01'):
    logger.info(f'mr_cluster start_date={start_date} end_date={end_date}')
    # all the hyper parameters
    cfg = dict(
        window_volume_wgt=60*24*30,
        window_volume_univ=60*24*20,

        windows_ewm=[5, 20, 100, 200, 800, 1000],
        windows_macd=[[5, 20], [20, 100], [100, 200], [200, 800]],
        windows_macd_signal=[[5, 20, 9], [20, 100, 30], [100, 200, 100]],

        windows_sharpe=[2000],

        # forward
        windows_forward=[10, 50],

        # kmeans config
        kmeans_lookback=24*60*30,
        kmeans_fitfreq=24*60*10,
        kmeans_k=20,

        # univ config
        universe_count=100,
        
        # applyops
        window_appops=1000,
    )

    featd = prepare_klines(start_date=start_date,
                           end_date=end_date,
                           folder=g_folder,
                           name=None,
                           r=g_r,
                           cfg=cfg)
    featd = define_univ(featd,
                        folder=g_folder,
                        name=None,
                        r=g_r,
                        cfg=cfg)

    # Very important to put a weight of 0 when outside of the universe
    featd['wgt'] = featd['wgt']*featd['univ']

    # features on the taker field in klines
    featd = klines_takerpct(featd,
                            folder=g_folder,
                            name=None,
                            r=g_r,
                            cfg=cfg)

    # turnover features
    featd = excess_turnover(featd,
                            folder=g_folder,
                            name=None,
                            r=g_r,
                            cfg=cfg)

    # Bumping return by the excess turnover
    featd['turnover_excess'] = featd['turnover_macdr20x100']
    featd['tret_xturnover'] = featd['turnover_excess']*featd['tret']
    featd['tret_xmkt_xturnover'] = featd['turnover_excess']*featd['tret_xmkt']

    # running the kmeans
    featd = kmeans_sret(featd,
                        incol='tret_xmkt',
                        oucol='sret_kmeans',
                        folder=g_folder,
                        name=None,
                        r=g_r,
                        cfg=cfg)

    # momentum on sret
    featd = mom_feats(featd,
                      feats=['sret_kmeans'],
                      folder=g_folder,
                      name=None,
                      r=g_r,
                      cfg=cfg)

    # MR mual
    featd = mr_mual_feats(featd,
                          feats=['sret_kmeans'],
                          outname='mual',
                          folder=g_folder,
                          name=None,
                          r=g_r,
                          cfg=cfg)

    # P&L features
    featd = pnl_feats(featd,
                      incol='mual',
                      fretcol='tret_xmkt',
                      folder=g_folder,
                      name=None,
                      r=g_r,
                      cfg=cfg)

    # PfP features
    featd = pfp_feats(featd,
                      feats=['tret_xmkt'],
                      folder=g_folder,
                      name=None,
                      r=g_r,
                      cfg=cfg)

    # volatility features
    featd = volatility_feats(featd,
                             feats=['tret_xmkt'],
                             folder=g_folder,
                             name=None,
                             r=g_r,
                             cfg=cfg)

    # adding the forward return
    featd, nfeats = perform_lag_forward(featd=featd,
                                        feats=['tret_xmkt'],
                                        windows=[-1],
                                        folder=g_folder,
                                        name=None,
                                        r=g_r)
    featd = rename_key(featd, nfeats[0], 'forward_fh1')

    # adding forward return 50 units
    for window_forward in cfg.get('windows_forward'):
        featd, nfeats = perform_sma(featd,
                                    feats=['tret_xmkt'],
                                    windows=[window_forward],
                                    folder=g_folder,
                                    name=None,
                                    r=g_r)
        featd, nfeats = perform_lag_forward(featd=featd,
                                            feats=['tret_xmkt'],
                                            windows=[-1*window_forward],
                                            folder=g_folder,
                                            name=None,
                                            r=g_r)
        featd = rename_key(featd, nfeats[0], f'forward_fh{window_forward}')
    g_r.save()
    return featd


def main_model(featd):
    model_lookback = 60*24*10
    model_fitfreq = 60*24*10

    featd = filter_dict_to_univ(featd)

    from crptmidfreq.mllib.lgbm_lin_v1 import gen_lgbm_lin_v1
    featd, nfeats = perform_model(featd,
                                  feats=get_sigf_cols(featd),
                                  wgt='wgt',
                                  ycol='forward_fh1',
                                  folder=g_folder,
                                  name="None",
                                  lookback=model_lookback,
                                  minlookback=model_lookback,
                                  fitfreq=model_fitfreq,
                                  gap=1,
                                  model_gen=partial(gen_lgbm_lin_v1, n_samples=1_000_000),
                                  with_fit=True,
                                  r=g_r)
    featd = rename_key(featd, nfeats[0], 'sig_ml')
    g_r.save()
    return featd


def main_signal_naive(featd):
    ipo_burn = 60*24

    for col in get_sigf_cols(featd):
        ncol = col.replace('sigf_', 'sig_')
        featd = rename_key(featd, col, ncol)

    # removing post IPO   20days
    for col in get_sig_cols(featd):
        featd[col] = featd[col]*(featd['sigf_ipocnt'] > ipo_burn)
        featd[col] = featd[col]*featd['univ']

    return featd


def bktest(featd):
    stats = perform_bktest(featd, folder=g_folder, name="None")

    # Common Bucketplots
    featd, nfeats = perform_bucketplot(featd,
                                       xcols=['wgt'],
                                       ycols=['forward_fh1'],
                                       n_buckets=8,
                                       freq=int(60*24*5),
                                       folder=g_folder,
                                       name="None",
                                       r=g_r)


def load_features(name=''):
    df = pd.read_parquet(os.path.join(get_analysis_folder(), f'kmeans_manual_v1_{name}.pq'))
    featd = {col: df[col].values for col in df.columns}
    return featd


def main():
    clean_folder(g_folder)
    #dts = pd.date_range('2020-01-01', '2024-03-01', freq='2MS')
    dts = pd.date_range('2020-01-01', '2021-01-01', freq='1M')
    for i in range(len(dts)-1):
        start_dt = dts[i].strftime('%Y-%m-%d')
        end_dt = dts[i+1].strftime('%Y-%m-%d')
        featd = main_features(start_date=start_dt, end_date=end_dt)
        #featd = main_model(featd)
        featd = main_signal_naive(featd)
        # saving down
        save_features(featd, name=f'kmeans_features_{start_dt}_{end_dt}')
        save_signal(featd, name=f'kmeans_signal_{start_dt}_{end_dt}')
        bktest(featd=featd)


# ipython -i -m crptmidfreq.res.mr_cluster_v1.kmeans_manual_v1
if __name__ == '__main__':
    main()
