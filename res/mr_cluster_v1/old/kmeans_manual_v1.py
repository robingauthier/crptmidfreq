import pandas as pd
import os
from crptmidfreq.utils.common import rename_key
from functools import partial
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.strats import *
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import get_sig_cols, get_sigf_cols
from crptmidfreq.utils.common import filter_dict_to_univ
from crptmidfreq.utils.common import filter_dict_to_dscode
from crptmidfreq.utils.common import filter_dict_to_dts
from crptmidfreq.utils.common import save_features
from crptmidfreq.utils.common import save_signal
from crptmidfreq.utils.common import to_csv
from crptmidfreq.utils.univ import hardcoded_universe_1

logger = get_logger()

g_folder = 'res_kmeans_v1'
g_r = StepperRegistry()




def main_features(start_date='2025-03-01', end_date='2026-01-01'):
    logger.info(f'mr_cluster start_date={start_date} end_date={end_date}')
    # all the hyper parameters
    cfg = dict(
        window_volume_wgt=60*24*30,
        window_volume_univ=60*24*20,

        windows_ewm=[5, 20, 100, 200, 800],
        windows_macd=[[5, 20], [20, 100], [200, 800]],
        windows_macd_signal=[[5, 20, 9], [20, 100, 30], [100, 200, 100]],

        windows_sharpe=[2000],

        # forward
        windows_forward=[10],

        # kmeans config
        kmeans_lookback=24*60*5,  # this is in time units
        kmeans_fitfreq=24*60*3,  # this is in time units
        kmeans_k=20,
        svd_lookback=24*60*20,
        svd_fitfreq=60,
        svd_k=20,
        kmeans_or_svd_or_naive='svd',

        # univ config
        universe_count=100,

        # applyops
        window_appops=1000,
    )
    if cfg['hardcoded_universe']:
        cfg['kmeans_k'] = 2
        cfg['svd_k'] = 2
    # arguments always used
    defargs = {'folder': g_folder, 'name': None, 'r': g_r, 'cfg': cfg}

    # read the data from the DuckDB
    featd = prepare_klines(start_date=start_date,
                           end_date=end_date,
                           tokens=hardcoded_universe_1 if cfg['hardcoded_universe'] else 'all',
                           **defargs)

    # Universe definition
    # Very important to put a weight of 0 when outside of the universe
    if not cfg['hardcoded_universe']:
        featd = define_univ(featd,
                            **defargs)
    else:
        featd['univ'] = np.ones_like(featd['dtsi'])

    # forward_fh1 definition
    featd = define_forward_fh(featd,
                              incol='tret',
                              **defargs)

    if g_all_features:
        # features on the taker field in klines
        featd = klines_takerpct(featd,
                                **defargs)

        # turnover features
        featd = excess_turnover(featd,
                                **defargs)

        # Bumping return by the excess turnover
        featd['turnover_excess'] = featd['turnover_macdr20x100']
        featd['tret_xturnover'] = featd['turnover_excess']*featd['tret']
        featd['tret_xmkt_xturnover'] = featd['turnover_excess']*featd['tret_xmkt']

    # running the kmeans
    featd = kmeans_sret(featd,
                        incol='tret_xmkt',
                        oucol='sret_kmeans',
                        **defargs)
    if g_hardcoded_universe:
        featd['sret_kmeans'] = featd['tret_xmkt_raw_clipqtl']

    if True:
        # momentum on sret
        featd = mom_feats(featd,
                          feats=['sret_kmeans'],
                          **defargs)
    if g_all_features:
        # PfP features -- this is too slow for now ! 2 minutes for 1 month
        featd = pfp_feats(featd,
                          feats=['tret_xmkt'],
                          **defargs)

    # MR mual
    featd = mr_mual_feats(featd,
                          feats=['sret_kmeans'],
                          outname='mual',
                          **defargs)

    # P&L features
    featd = pnl_feats(featd,
                      incol='mual',
                      fretcol='tret_xmkt',
                      **defargs)

    if g_all_features:
        # volatility features
        featd = volatility_feats(featd,
                                 feats=['tret_xmkt'],
                                 **defargs)

    featd, _ = perform_clean_memory(featd)
    g_r.save()
    return featd


def main_model(featd):
    unit_day = 60*24
    model_lookback = unit_day*30  # I get into RAM issues otherwise
    model_fitfreq = unit_day*10

    featd = filter_dict_to_univ(featd)

    from crptmidfreq.mllib.lgbm_lin_v1 import gen_lgbm_lin_v1
    featd, nfeats = perform_model(featd,
                                  feats=get_sigf_cols(featd),
                                  wgt='wgt',
                                  ycol='forward_fh1',
                                  folder=g_folder,
                                  name="None",
                                  lookback=model_lookback,
                                  minlookback=int(0.2*model_lookback),
                                  fitfreq=model_fitfreq,
                                  gap=1,
                                  model_gen=partial(gen_lgbm_lin_v1, n_samples=1_000_000),
                                  with_fit=True,
                                  r=g_r)
    featd = rename_key(featd, nfeats[0], 'sig_ml')
    g_r.save()
    return featd


def main_signal_naive(featd):
    ipo_burn = 60*24

    for col in get_sigf_cols(featd):
        ncol = col.replace('sigf_', 'sig_')
        featd = rename_key(featd, col, ncol)

    # removing post IPO   20days
    for col in get_sig_cols(featd):
        #featd[col] = featd[col]*(featd['sigf_ipocnt'] > ipo_burn)
        featd[col] = featd[col]*featd['univ']

    # for bench
    featd['sig_one'] = np.ones_like(featd['wgt'])
    return featd


def bktest(featd):
    stats = perform_bktest(featd, folder=g_folder, name="None")

    # Common Bucketplots
    featd, nfeats = perform_bucketplot(featd,
                                       xcols=['wgt'],
                                       ycols=['forward_fh1'],
                                       n_buckets=8,
                                       freq=int(60*24*5),
                                       folder=g_folder,
                                       name="None",
                                       r=g_r)


def dump_extract(featd):
    """Dumping the data for manual checks"""
    logger.info('Dumping the data for manual checks')

    icols = ['dtsi', 'dscode_str', 'close', 'sig_zs_100', 'sig_mual',
             'univ', 'kmeans_cat', 'turnover_excess']
    df = pd.DataFrame({k: v for k, v in featd.items() if k in icols})
    df['dtsi'] = pd.to_datetime(df['dtsi']*1e3)

    df_cs_loc = df[df['dtsi'] == df['dtsi'].iloc[-3000]]
    to_csv(df_cs_loc, 'dfnice_cs_loc')

    df_ts_loc = df[df['dscode_str'] == 'BCHUSDT']
    to_csv(df_ts_loc, 'dfnice_ts_bchusdt')

    df_g = df.assign(cnt=1)\
        .groupby('dtsi')\
        .agg({'sig_mual': ('mean', 'std'), 'univ': 'sum', 'cnt': 'sum'})
    to_csv(df_g, 'df_univ_cnt')

    # all the columns now
    featd_loc = filter_dict_to_dscode(featd, dscode_str='BCHUSDT')
    df_loc = pd.DataFrame(featd_loc)
    df_loc['dtsi'] = pd.to_datetime(df_loc['dtsi']*1e3)
    to_csv(df_loc, 'df_allcols_bchusdt')

    dtsi = np.max(featd['dtsi'])
    featd_loc = filter_dict_to_dts(featd, dtsi=dtsi)
    df_loc = pd.DataFrame(featd_loc)
    df_loc['dtsi'] = pd.to_datetime(df_loc['dtsi']*1e3)
    to_csv(df_loc, 'df_cs_loc_allcols')


def main():
    clean_folder(g_folder)
    dts = pd.date_range('2023-01-01', '2025-05-01', freq='1MS')
    #dts = pd.date_range('2024-01-01', '2025-05-01', freq='5D')
    #dts = pd.date_range('2020-01-01', '2021-01-01', freq='1M')
    for i in range(len(dts)-1):
        start_dt = dts[i].strftime('%Y-%m-%d')
        end_dt = dts[i+1].strftime('%Y-%m-%d')
        featd = main_features(start_date=start_dt, end_date=end_dt)
        featd = main_model(featd)
        #featd = main_signal_naive(featd)
        # return featd
        # dump_extract(featd)
        # saving down
        save_features(featd, name=f'kmeans_features_{start_dt}_{end_dt}')
        save_signal(featd, name=f'kmeans_signal_{start_dt}_{end_dt}')
        bktest(featd=featd)


# ipython -i -m crptmidfreq.res.mr_cluster_v1.kmeans_manual_v1
if __name__ == '__main__':
    featd = main()
