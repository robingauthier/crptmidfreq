
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


def mom_feats(featd, feats=['sret_kmeans'], folder=None, name=None, r=None, cfg={}):

    dcfg = dict(
        windows_ewm=[5, 20, 100, 500, 1000],
        windows_macd=[[5, 20], [20, 100], [100, 500]],
        windows_macd_signal=[[5, 20, 9], [20, 100, 30], [100, 200, 100]],
        window_appops=1000
    )
    cfg = merge_dicts(cfg, dcfg, name='klines_takerpct')

    # macd
    featd, nfeats1 = perform_macd(featd=featd,
                                  feats=feats,
                                  windows=cfg.get('windows_macd'),
                                  folder=folder,
                                  name=name,
                                  r=r)

    # macd signal line
    featd, nfeats2 = perform_macd_signal(featd=featd,
                                         feats=feats,
                                         windows=cfg.get('windows_macd_signal'),
                                         folder=folder,
                                         name=name,
                                         r=r)

    # ewm(X)/std(X)
    featd, nfeats3 = perform_ewm_scaled(featd=featd,
                                        feats=feats,
                                        windows=cfg.get('windows_ewm'),
                                        folder=folder,
                                        name=name,
                                        r=r)

    # computing the skew and kurtosis of the sret
    featd, nfeats_skew = perform_ewm_skew(featd,
                                          feats=feats,
                                          windows=cfg.get('windows_ewm'),
                                          folder=folder,
                                          name=name,
                                          r=r)

    featd, nfeats_kurt = perform_ewm_kurt(featd,
                                          feats=feats,
                                          windows=cfg.get('windows_ewm'),
                                          folder=folder,
                                          name=name,
                                          r=r)

    nfeats = nfeats1+nfeats2+nfeats3+nfeats_skew+nfeats_kurt
    # apply_ops step now
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
