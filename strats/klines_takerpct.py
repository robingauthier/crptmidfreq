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


def klines_takerpct(featd, folder=None, name=None, r=None, cfg={}):
    dcfg = dict(
        windows_ewm=[5, 20, 100, 500, 1000],
        windows_macd=[[5, 20], [20, 100], [100, 500]],
    )
    cfg = merge_dicts(cfg, dcfg, name='klines_takerpct')

    # taker_buy_turnover/turnover
    featd, nfeats = perform_divide(featd,
                                   numcols=['taker_buy_volume'],
                                   denumcols=['volume'],
                                   folder=folder,
                                   name=name,
                                   r=r)
    featd['takerpct'] = featd[nfeats[0]]
    featd['takerpct'] = featd['takerpct'] - 0.5
    # ewm(X)
    featd, nfeats1 = perform_ewm(featd=featd,
                                 feats=nfeats,
                                 windows=cfg.get('windows_ewm'),
                                 folder=folder,
                                 name=name,
                                 r=r)
    # macd now
    featd, nfeats2 = perform_macd(featd=featd,
                                  feats=nfeats,
                                  windows=cfg.get('windows_macd'),
                                  folder=folder,
                                  name=name,
                                  r=r)

    # ewm(X)/std(X)
    featd, nfeats3 = perform_ewm_scaled(featd=featd,
                                        feats=nfeats,
                                        windows=cfg.get('windows_ewm'),
                                        folder=folder,
                                        name=name,
                                        r=r)

    # Cross sectional step now
    featd, _ = perform_to_sigf(featd,
                               nfeats1+nfeats2+nfeats3,
                               folder=folder,
                               name=name,
                               r=r)

    return featd
