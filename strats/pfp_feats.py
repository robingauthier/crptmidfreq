from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts


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
