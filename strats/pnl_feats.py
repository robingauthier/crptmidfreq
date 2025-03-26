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
from crptmidfreq.strats.prepare_klines import prepare_klines
from crptmidfreq.utils.common import merge_dicts


def pnl_feats(featd, incol='mual', fretcol='tret_xmkt', 
              folder=None, name=None, r=None, cfg={}):
    assert fretcol in featd.keys()
    dcfg = dict(
        windows_sharpe=[1000, 5000],
        window_appops=3000
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')

    # computing the P&L
    featd, nfeats = perform_lag(featd,
                                feats=[incol],
                                windows=[1],
                                folder=folder,
                                name=name,
                                r=r)
    featd[f'{incol}_pnl'] = featd[incol]*featd['tret_xmkt']

    # computing the sharpe of the mual
    nfeats_sh = []
    for shwin in cfg.get('windows_sharpe'):
        featd, nfeat_sh = perform_sharpe_ewm(featd,
                                             sigcol=incol,
                                             pnlcol=f'{incol}_pnl',
                                             window=shwin,
                                             folder=folder,
                                             name=name,
                                             r=r)
        nfeats_sh += nfeat_sh

        # Computing a quantile on sharpe
        featd, nfeat_sharpe_qtl = perform_quantile_global(featd,
                                                          feats=nfeat_sh,
                                                          qs=[0.4],
                                                          folder=folder,
                                                          name=name,
                                                          r=r)
        featd[f'sigf_{incol}_high_sharpe{shwin}'] = featd[incol]*(featd[nfeat_sh[0]] > featd[nfeat_sharpe_qtl[0]])

    # Computing the drawdown
    feats, nfeatdd = perform_drawdown(featd,
                                      sigcol=incol,
                                      pnlcol=f'{incol}_pnl',
                                      folder=folder,
                                      name=name,
                                      r=r)

    # Computing a quantile on drawdown
    featd, nfeats_dd_qtl = perform_quantile_global(featd,
                                                   feats=nfeatdd,
                                                   qs=[0.4],
                                                   folder=folder,
                                                   name=name,
                                                   r=r)
    featd[f'sigf_{incol}_high_sharpe'] = featd[incol]*(featd[nfeatdd[0]] > featd[nfeats_dd_qtl[0]])

    # apply_ops step
    nfeats = nfeatdd + nfeats_sh
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
