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
from crptmidfreq.utils.common import merge_dicts
from crptmidfreq.utils.common import get_sig_cols, get_sigf_cols, get_forward_cols
from crptmidfreq.utils.common import ewm_alpha
from crptmidfreq.strats.prepare_klines import prepare_klines


def excess_turnover(featd, folder=None, name=None, r=None, cfg={}):
    assert 'turnover' in featd.keys()
    dcfg = dict(
        windows_macd=[[5, 20], [20, 100], [100, 500]],
        window_appops=3000,
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')

    # adding the excess turnover
    featd, nfeats = perform_macd_ratio(featd,
                                       feats=['turnover'],
                                       windows=cfg.get('windows_macd'),
                                       folder=folder,
                                       name=name,
                                       r=r)

    featd, nfeatcs = perform_cs_appops(featd,
                                       feats=nfeats,
                                       windows=[cfg.get('window_appops')],
                                       folder=folder,
                                       name=name,
                                       r=r)

    featd, _ = perform_to_sigf(featd,
                               feats=nfeatcs,
                               folder=folder,
                               name=name,
                               r=r)
    return featd
