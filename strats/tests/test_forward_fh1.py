import pandas as pd

from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.strats import *

# pytest ./crptmidfreq/strats/tests/test_forward_fh1.py --pdb --maxfail=1


g_folder = 'res_test_v1'
g_r = StepperRegistry()


def test_forward_fh1(start_date='2025-01-10', end_date='2025-01-13'):
    """
    - you want Sum of forward_fh1*wgt to be 0 on a daily basis
    - 
    """

    logger.info(f'mr_cluster start_date={start_date} end_date={end_date}')
    # all the hyper parameters
    cfg = dict(
        window_volume_wgt=60*24*30,
        window_volume_univ=60*24*20,
        # univ config
        universe_count=100,
    )
    featd = prepare_klines(start_date=start_date,
                           end_date=end_date,
                           folder=g_folder,
                           name=None,
                           r=g_r,
                           cfg=cfg)

    # Universe definition
    # Very important to put a weight of 0 when outside of the universe
    featd = define_univ(featd,
                        folder=g_folder,
                        name=None,
                        r=g_r,
                        cfg=cfg)

    # Preparing the forward return now
    featd, nfeats = perform_lag_forward(featd=featd,
                                        feats=['tret'],
                                        windows=[-1],
                                        folder=g_folder,
                                        name=None,
                                        r=g_r)

    # uses wgt to remove the market
    featd = remove_mkt(featd,
                       incol='forward_tret_lag-1',
                       outcol='forward_fh1',
                       with_clip=False,
                       folder=g_folder,
                       name=None,
                       r=g_r,
                       cfg=cfg)

    # we want to check this
    featd['sig_one'] = np.ones_like(featd['wgt'])

    cls_bk = perform_bktest(featd, folder=g_folder, name="None")

    # Check that the sharpe is 0.0
    assert abs(cls_bk.statsdf['sr'].iloc[0]) < 0.1

    icols = ['dtsi', 'dscode_str', 'close', 'wgt', 'univ', 'forward_fh1', 'tret']
    df = pd.DataFrame({k: v for k, v in featd.items() if k in icols})
    df['dts'] = pd.to_datetime(df['dtsi']*1e3)

    df['forward_fh1_wgt'] = df['forward_fh1']*df['wgt']
    dfg = df.groupby('dtsi').agg({'forward_fh1_wgt': 'mean'})
    assert dfg['forward_fh1_wgt'].abs().mean() < 1e-4
    assert dfg['forward_fh1_wgt'].abs().max() < 1e-4

    # check that BTCUSDT is part of the universe !
    assert df[df['dscode_str'] == 'BTCUSDT']['univ'].mean() > 0.95
    assert df[df['dscode_str'] == 'ETHUSDT']['univ'].mean() > 0.95
    assert df[df['dscode_str'] == 'XRPUSDT']['univ'].mean() > 0.95
    assert df[df['dscode_str'] == 'BNBUSDT']['univ'].mean() > 0.95
