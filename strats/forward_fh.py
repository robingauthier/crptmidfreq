from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.strats import *
from crptmidfreq.utils.common import get_logger, merge_dicts, rename_key

logger = get_logger()


def define_forward_fh(featd, incol='tret', folder=None, name=None, r=None, cfg={}):
    """
    input on : incol
    output is all the forward_fh1 ,... 
    as well as {incol}_xmkt and {incol}_xmkt_clip
    """
    assert incol in featd.keys()
    assert 'wgt' in featd.keys()
    dcfg = dict(
        windows_forward=[10],
        forward_xmkt=True,
    )
    cfg = merge_dicts(cfg, dcfg, name='kmeans_sret')
    defargs = {'folder': folder, 'r': r, 'name': name}

    # defining tret_xmkt
    featd = remove_mkt(featd,
                       incol=incol,
                       outcol=f'{incol}_xmkt',
                       with_clip=True,
                       **defargs)

    # Preparing the forward return now
    featd, nfeats = perform_lag_forward(featd=featd,
                                        feats=[incol],
                                        windows=[-1],
                                        **defargs)

    if cfg.get('forward_xmkt'):
        # uses wgt to remove the market
        featd = remove_mkt(featd,
                           incol=f'forward_{incol}_lag-1',
                           outcol='forward_fh1',
                           with_clip=False,
                           **defargs)
    else:
        featd['forward_fh1'] = featd[f'forward_{incol}_lag-1']

    # forward_fh1_clip
    featd, nfeats = perform_clip_quantile_global(featd,
                                                 feats=['forward_fh1'],
                                                 low_clip=0.03,
                                                 high_clip=0.97,
                                                 folder=folder,
                                                 name=name,
                                                 r=r)
    featd = rename_key(featd, nfeats[0], 'forward_fh1_clip')

    # adding forward return X units
    for window_forward in cfg.get('windows_forward'):
        featd, nfeats = perform_sma(featd,
                                    feats=['tret'],
                                    windows=[window_forward],
                                    **defargs)
        featd, nfeats = perform_lag_forward(featd=featd,
                                            feats=nfeats,
                                            windows=[-1*window_forward],
                                            **defargs)
        if cfg.get('forward_xmkt'):
            featd = remove_mkt(featd,
                               incol=nfeats[0],
                               outcol=f'forward_fh{window_forward}',
                               with_clip=False,
                               **defargs)
        else:
            featd[f'forward_fh{window_forward}'] = featd[nfeats[0]]

        # forward_fhX_clip
        featd, nfeats = perform_clip_quantile_global(featd,
                                                     feats=[f'forward_fh{window_forward}'],
                                                     low_clip=0.03,
                                                     high_clip=0.97,
                                                     folder=folder,
                                                     name=name,
                                                     r=r)
        featd = rename_key(featd, nfeats[0], f'forward_fh{window_forward}_clip')
    return featd
