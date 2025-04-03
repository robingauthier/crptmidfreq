
import numpy as np
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts


def gap_feats_loc(featd, incol='sret', folder=None, name=None, r=None, cfg={}):
    dcfg = dict(
        gaps_lag=1,
        gaps_window=1000,
        gaps_th=2.0,
        gaps_ewm_windows=[10, 100],
    )
    cfg = merge_dicts(cfg, dcfg, name='gap_feats')
    defargs = {'folder': folder, 'name': name, 'r': r}

    # Computing a risk
    featd, feats_std = perform_ewm_std(featd,
                                       feats=[incol],
                                       windows=[cfg.get('gaps_window')],
                                       **defargs)

    # Computing the price
    featd, feats_px = perform_cumsum(featd,
                                     feats=[incol],
                                     **defargs)

    # Computing max and min lagged by 1
    featd, feats_max0 = perform_rolling_max(featd,
                                            feats=feats_px,
                                            windows=[cfg.get('gaps_window')],
                                            **defargs)
    featd, feats_max = perform_lag(featd,
                                   feats=feats_max0,
                                   windows=[cfg.get('gaps_lag')],
                                   **defargs)
    featd, feats_min0 = perform_rolling_min(featd,
                                            feats=feats_px,
                                            windows=[cfg.get('gaps_window')],
                                            **defargs)
    featd, feats_min = perform_lag(featd,
                                   feats=feats_min0,
                                   windows=[cfg.get('gaps_lag')],
                                   **defargs)

    # Gap definition
    gaps_th = cfg.get('gaps_th')
    gaps_window = cfg.get('gaps_window')
    gaps_lag = cfg.get('gaps_lag')
    gaps_th = cfg.get('gaps_th')
    # px > (1 + th* risk) * pxRollMax
    featd['gap_up_evt'] = 1.0*(featd[feats_px[0]] >
                               (1+gaps_th*np.sqrt(gaps_lag)*featd[feats_std[0]])*featd[feats_max[0]])
    # px < (1 - th* risk) * pxRollMin
    featd['gap_dn_evt'] = 1.0*(featd[feats_px[0]] <
                               (1-gaps_th*np.sqrt(gaps_lag)*featd[feats_std[0]])*featd[feats_min[0]])
    col = f'gap_{gaps_window}x{gaps_lag}x{gaps_th}'
    featd[col] = featd['gap_up_evt']-featd['gap_dn_evt']

    featd, feats_ewm = perform_ewm(featd,
                                   feats=[col],
                                   windows=cfg.get('gaps_ewm_windows'),
                                   **defargs)

    featd, _ = perform_to_sigf(featd,
                               feats=feats_ewm,
                               **defargs)
    return featd


def gap_feats(featd, incol='sret', folder=None, name=None, r=None, cfg={}):
    dcfg = dict(
        gaps_lag=1,
        gaps_windows=[1000],
        gaps_ths=[2.0],
        gaps_ewm_windows=[10, 100],
    )
    cfg = merge_dicts(cfg, dcfg, name='gap_feats')
    defargs = {'folder': folder, 'name': name, 'r': r}
    for gap_window in cfg.get('gaps_windows'):
        for gap_th in cfg.get('gaps_ths'):
            ncfg = {
                'gaps_lag': cfg['gaps_lag'],
                'gaps_window': gap_window,
                'gaps_th': gap_th,
                'gaps_ewm_windows': cfg['gaps_ewm_windows'],
            }
            # Call the lower-level feature function with full config
            featd = gap_feats_loc(
                featd,
                incol=incol,
                folder=defargs['folder'],
                name=defargs['name'],
                r=defargs['r'],
                cfg=ncfg
            )
    return featd
