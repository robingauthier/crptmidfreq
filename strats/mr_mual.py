
from crptmidfreq.utils.common import rename_key
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import ewm_alpha
from crptmidfreq.utils.common import merge_dicts


def mr_mual_feats(featd, feats=['sret_kmeans'], outname='mual', folder=None, name=None, r=None, cfg={}):
    check_cols(featd, ['wgt'])
    assert len(feats) == 1
    dcfg = dict(
        windows_ewm=[5, 20, 100, 500, 1000],
    )
    cfg = merge_dicts(cfg, dcfg, name='mr_mual_feats')

    # computing different ewms of the residuals
    featd, nfeats_ewm_sr = perform_ewm(featd=featd,
                                       feats=feats,
                                       windows=cfg.get('windows_ewm'),
                                       folder=folder,
                                       name=name,
                                       r=r)

    # mual_std :: computing the volatility of the sret_kmeans
    featd, nfeats_vol_sr = perform_ewm_std(featd=featd,
                                           feats=feats,
                                           windows=cfg.get('windows_ewm'),
                                           folder=folder,
                                           name=name,
                                           r=r)
    featd, _ = perform_avg_features_fillna0(
        featd,
        xcols=nfeats_vol_sr,
        outname=f'{outname}_std',
        folder=folder,
        name=name,
        r=r)

    # Manually computing the ewm(X)/mual_std
    nfeats_zs = []
    nwins = len(cfg.get('windows_ewm'))
    for i in range(nwins):
        ewm_col = nfeats_ewm_sr[i]
        win = cfg.get('windows_ewm')[i]
        alpha = 1-ewm_alpha(win)

        # ewm(X) has Var = Var(x_i)* (1-alpha)/(1+alpha)
        featd[f'todel_{ewm_col}'] = featd[ewm_col]*np.sqrt((1+alpha)/(1-alpha))
        featd, nfeats = perform_divide(featd,
                                       numcols=[f'todel_{ewm_col}'],
                                       denumcols=[f'{outname}_std'],
                                       folder=folder,
                                       name=name,
                                       r=r)
        featd, nfeats = perform_clip(featd=featd,
                                     feats=nfeats,
                                     low_clip=-3.0,
                                     high_clip=3.0,
                                     folder=folder,
                                     name=name,
                                     r=r)

        # adjusting for the cross sectional standard deviation now
        # featd, feats=[], by=None, wgt=None, folder=None, name=None, r=g_reg
        featd, nfeats = perform_cs_scaling(featd=featd,
                                           feats=nfeats,
                                           by=None,
                                           wgt='wgt',
                                           folder=folder,
                                           name=name,
                                           r=r)

        featd = rename_key(featd, nfeats[0], f'zs_{win}')
        nfeats_zs += [f'zs_{win}']

    # Here we create the column mual as the average of all the above
    featd, _ = perform_avg_features_fillna0(featd,
                                            xcols=nfeats_zs,
                                            outname=f'{outname}_temp',
                                            folder=folder,
                                            name=name,
                                            r=r)
    featd, nfeats = perform_clip(featd=featd,
                                 feats=[f'{outname}_temp'],
                                 low_clip=-3.0,
                                 high_clip=3.0,
                                 folder=folder,
                                 name=name,
                                 r=r)
    featd = rename_key(featd, nfeats[0], outname)

    # Changing the signs now
    for col in nfeats_zs+[outname]:
        featd[col] = -1.0*featd[col]

    # All the zs will be features
    featd, _ = perform_to_sigf(featd,
                               feats=nfeats_zs,
                               folder=folder,
                               name=name,
                               r=r)
    featd[f'sigf_{outname}'] = featd[outname]
    return featd
