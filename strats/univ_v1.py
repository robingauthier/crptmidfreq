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

logger = get_logger()


def define_univ(featd, folder=None, name=None, r=None, cfg={}):
    assert 'turnover' in featd.keys()
    dcfg = dict(
        window_volume_univ=60*24*20,
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')

    # rank cross sectionally by volume to build a robust universe
    featd, nfeats_ev = perform_ewm(featd=featd,
                                   feats=['turnover'],
                                   windows=[cfg.get('window_volume_univ')],
                                   folder=folder,
                                   name=name,
                                   r=r)
    featd = rename_key(featd, nfeats_ev[0], 'turnover_ewm')
    featd, nfeats = perform_cs_rank_int_decreasing(featd=featd,
                                                   feats=['turnover_ewm'],
                                                   folder=folder,
                                                   name=name,
                                                   r=r)
    featd['sigf_turnover_rank'] = featd[nfeats[0]]

    # define universe
    featd['univ'] = 1*(featd[nfeats[0]] <= cfg.get('universe_count'))
    return featd
