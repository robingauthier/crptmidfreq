
import pandas as pd
import os
import duckdb
from crptmidfreq.utils.common import rename_key
from crptmidfreq.config_loc import get_data_db_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import merge_dicts


logger = get_logger()


def remove_mkt(featd, incol='tret', outcol='tret_xmkt', with_clip=True, folder=None, name=None, r=None, cfg={}):
    dcfg = dict(
        tret_clip_pct=0.02,
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')

    # Removing the market in tret => tret_xmkt
    featd, nfeats = perform_cs_demean(featd=featd,
                                      feats=[incol],
                                      by=None,
                                      wgt='wgt',
                                      folder=folder,
                                      name=name,
                                      r=r)
    if with_clip:
        featd = rename_key(featd, nfeats[0], f'{outcol}_raw')
        th = cfg.get('tret_clip_pct')
        featd, nfeats = perform_clip_quantile_global(featd=featd,
                                                     feats=[f'{outcol}_raw'],
                                                     low_clip=th,
                                                     high_clip=1-th,
                                                     folder=folder,
                                                     name=name,
                                                     r=r)
        featd[outcol] = featd[nfeats[0]]
    else:
        featd[outcol] = featd[nfeats[0]]

    # forward_fh1 must be 0 outside univ
    if 'univ' in featd.keys():
        featd[outcol] = featd[outcol]*featd['univ']
    featd[outcol] = np.where(featd['wgt'] == 0.0, 0.0, featd[outcol])

    return featd
