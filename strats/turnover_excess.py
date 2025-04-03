from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts


def turnover_excess(featd, folder=None, name=None, r=None, cfg={}):
    """
    pdf=df.pivot_table(index='close_time',columns='dscode_str',values='turnover_log')
    pdf = pdf.diff().fillna(0.0)
    avg_pdf=pdf.mean(axis=1)
    for col in pdf.columns:
        pdf[col]=pdf[col]-avg_pdf
    pdf = pdf.cumsum()
    pdf=pdf.ewm(halflife=10000,min_periods=1000).mean()-pdf.ewm(halflife=100000,min_periods=1000).mean()
    pdf.plot()
    """
    assert 'turnover' in featd.keys()
    unit_day = 60*24
    dcfg = dict(
        windows_macd_turnover=[[5*unit_day, 30*unit_day]],
        window_appops=3000,
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')
    defargs = {'folder': folder, 'name': name, 'r': r}
    featd['turnover_log'] = np.log(1.0+np.nan_to_num(featd['turnover']))

    # we rebuild a turnover cross sectionally adjusted
    featd, nfeats_diff = perform_diff(featd,
                                      feats=['turnover_log'],
                                      windows=[1],
                                      **defargs)
    featd, nfeat_cs = perform_cs_demean(featd,
                                        feats=nfeats_diff,
                                        **defargs)

    featd, nfeat_nturn = perform_cumsum(featd,
                                        feats=nfeat_cs,
                                        **defargs)
    # Now we have our new turnover. We just need some macd
    featd, rfeats = perform_macd(featd,
                                 feats=nfeat_nturn,
                                 windows=cfg.get('windows_macd_turnover'),
                                 **defargs)

    featd, _ = perform_to_sigf(featd,
                               feats=rfeats,
                               **defargs)

    return featd
