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
from crptmidfreq.strats.univ_v1 import define_univ


def pfp_feats(featd, feats=['tret_xmkt'], folder=None, name=None, r=None, cfg={}):
    dcfg = dict(
        pfp_nbrevs=[3],
        pfp_ticks=[0.5, 1, 2, 5],  # we work on px //tick
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')

    featd, feats_log = perform_log(featd,
                                   feats=feats,
                                   folder=folder,
                                   name=name,
                                   r=r)
    featd, feats_px = perform_cumsum(featd,
                                     feats=feats_log,
                                     folder=folder,
                                     name=name,
                                     r=r)
    for col in feats_px:
        featd[col] = 1+featd[col]

    # pfp operates on prices, or I1 series
    # taking the log and then cumsum
    featd, nfeats = perform_pfp(featd,
                                feats=feats_px,
                                nbrevs=cfg.get('pfp_nbrevs'),
                                ticks=cfg.get('pfp_ticks'),
                                debug=False,
                                folder=folder,
                                name=name,
                                r=r)

    featd, _ = perform_to_sigf(featd,
                               feats=nfeats,
                               folder=folder,
                               name=name,
                               r=r)
    return featd
