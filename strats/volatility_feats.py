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
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import get_sig_cols, get_sigf_cols, get_forward_cols
from crptmidfreq.utils.common import ewm_alpha
from crptmidfreq.utils.common import merge_dicts
from crptmidfreq.strats.prepare_klines import prepare_klines



def volatility_feats(featd, feats=['tret_xmkt'], folder=None, name=None, r=None, cfg={}):
    check_cols(featd, feats)
    dcfg = dict(
        windows_macd=[[5, 20], [20, 100], [100, 500]],
        window_appops=3000,
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')

    # computing volatility
    infer_windows = []
    for wins in cfg.get('windows_macd'):
        infer_windows += wins
    infer_windows = set(infer_windows)
    featd, _ = perform_ewm_std(featd,
                               feats=feats,
                               windows=cfg.get('windows_macd'),
                               folder=folder,
                               name=name,
                               r=r)

    # macdratio on those volatilites
    nfeats = []
    for col in feats:
        for hls in cfg.get('windows_macd'):
            hlfast = hls[0]
            hlslow = hls[1]
            denum = featd[f'{col}_ewmstd{hlslow}']
            featd[f'{col}_ewmstd_macdr{hlfast}x{hlslow}'] = np.divide(
                featd[f'{col}_ewmstd{hlfast}'],
                denum,
                out=np.ones_like(denum),
                where=~np.isclose(denum,
                                  np.zeros_like(denum)))
            nfeats += [f'{col}_ewmstd_macdr{hlfast}x{hlslow}']

    # Cross sectional ranking
    featd, nfeatcs = perform_cs_appops(featd,
                                       feats=nfeats,
                                       windows=[cfg.get('window_appops')],
                                       folder=folder,
                                       name=name,
                                       r=r)

    featd, _ = perform_to_sigf(featd,
                               feats=nfeatcs+nfeats,
                               folder=folder,
                               name=name,
                               r=r)
    return featd
