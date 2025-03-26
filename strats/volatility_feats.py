from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts


def volatility_feats(featd, feats=['tret_xmkt'], folder=None, name=None, r=None, cfg={}):
    """
    these are measuring changes in volatility
    TODO we need to check if same as  change(vol_5d ) ? 
    """
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
    infer_windows = list(set(infer_windows))
    featd, _ = perform_ewm_std(featd,
                               feats=feats,
                               windows=infer_windows,
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
