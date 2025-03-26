from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts


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
