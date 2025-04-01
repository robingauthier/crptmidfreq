import pandas as pd
import os
from crptmidfreq.utils.common import rename_key
from pprint import pprint
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
from crptmidfreq.mllib.lgbm_lin_v1 import gen_lgbm_lin_v1
from crptmidfreq.mllib.feedforward_v1 import gen_feed_forward
from crptmidfreq.mllib.boosting_torch import gen_boosting_torch
from crptmidfreq.mllib.linear_torch import gen_linear_torch
from crptmidfreq.utils.univ import hardcoded_universe_1


# This will give you intuitions as to what kind of strategies should work

logger = get_logger()

g_folder = 'res_kmeans_naive_v1'
g_r = StepperRegistry()


def main_features(start_date='2025-03-01', end_date='2026-01-01'):
    logger.info(f'mr_cluster start_date={start_date} end_date={end_date}')
    # all the hyper parameters
    unit_day = 60*24

    cfg = dict(
        use_lazy_dict=True,

        window_volume_wgt=60*24*30,
        window_volume_univ=60*24*20,

        windows_ewm=[5, 20, 100, 200, 800],

        windows_forward=[10],
        forward_xmkt=True,  # removes mkt from forward return TODO True

        sret_clip=0.005,  # we should not cut too much

        # kmeans config
        kmeans_lookback=unit_day*5,  # this is in time units
        kmeans_fitfreq=unit_day*3,  # this is in time units
        kmeans_k=20,
        svd_lookback=unit_day*20,
        svd_fitfreq=60,
        svd_k=20,
        kmeans_or_svd_or_naive='naive',

        # univ config
        universe_count=100,
        hardcoded_universe=False,  # TODO: roll back

        # applyops
        window_appops=1000,

        nb_fetures=1,  # 1,2  1 is minimum amount of features

        model_lookback=unit_day*200,  # I get into RAM issues otherwise
        model_minlookback=unit_day*10,
        model_fitfreq=unit_day*20,

        ml_kind='mlpytorch',

    )
    if cfg['hardcoded_universe']:
        cfg['kmeans_k'] = 2
        cfg['svd_k'] = 2
    defargs = {'folder': g_folder, 'name': None, 'r': g_r, 'cfg': cfg}
    pprint(cfg)

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
    # and tret_xmkt definition
    featd = define_forward_fh(featd,
                              incol='tret',
                              **defargs)

    # running the kmeans
    kind = cfg.get('kmeans_or_svd_or_naive')
    if kind == 'kmeans':
        featd = kmeans_sret(featd,
                            incol='tret_xmkt',
                            oucol='sret',
                            **defargs)
    elif kind == 'svd':
        featd = svd_sret(featd,
                         incol='tret_xmkt',
                         oucol='sret',
                         **defargs)
    else:
        featd['sret'] = featd['tret_xmkt']  # it is clipped yes

    # MR mual
    featd = mr_mual_feats(featd,
                          feats=['sret'],
                          outname='mual',
                          **defargs)

    if cfg.get('nb_fetures') > 1:
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

        # momentum on sret
        featd = mom_feats(featd,
                          feats=['sret'],
                          **defargs)

        # PfP features -- this is too slow for now ! 2 minutes for 1 month
        featd = pfp_feats(featd,
                          feats=['tret_xmkt'],
                          **defargs)

        # P&L features
        featd = pnl_feats(featd,
                          incol='mual',
                          fretcol='tret_xmkt',
                          **defargs)

        # volatility features
        featd = volatility_feats(featd,
                                 feats=['tret_xmkt'],
                                 **defargs)

    if cfg.get('ml_kind') == 'no_ml':
        # Converting sigf to sig
        for col in get_sigf_cols(featd):
            ncol = col.replace('sigf_', 'sig_')
            featd = rename_key(featd, col, ncol)
    elif cfg.get('ml_kind') == 'ml':
        featd = filter_dict_to_univ(featd)
        featd, nfeats_ml = perform_model(featd,
                                         feats=get_sigf_cols(featd),
                                         wgt='wgt',
                                         ycol='forward_fh1_clip',
                                         folder=g_folder,
                                         name="None",
                                         lookback=cfg.get('model_lookback'),
                                         minlookback=cfg.get('model_minlookback'),
                                         fitfreq=cfg.get('model_fitfreq'),
                                         gap=1,
                                         model_gen=partial(gen_lgbm_lin_v1, n_samples=1_000_000),
                                         with_fit=True,
                                         r=g_r)
    elif cfg.get('ml_kind') == 'mlpytorch':
        featd = filter_dict_to_univ(featd)
        featd, nfeats_ml = perform_model_batch(featd,
                                               feats=get_sigf_cols(featd),
                                               wgt='wgt',
                                               ycol='forward_fh1_clip',
                                               folder=g_folder,
                                               name="None",
                                               lookback=cfg.get('model_lookback'),
                                               minlookback=cfg.get('model_minlookback'),
                                               ramlookback=1*unit_day,
                                               batch_size=300,
                                               epochs=10,
                                               lr=1e-4,
                                               weight_decay=1e-2,
                                               fitfreq=cfg.get('model_fitfreq'),
                                               gap=1,
                                               model_gen=gen_linear_torch,
                                               #model_gen=partial(gen_feed_forward, hidden_dim=20),
                                               #model_gen=partial(gen_boosting_torch, hidden_dim=20),
                                               with_fit=True,
                                               r=g_r)
    else:
        raise(ValueError('issue ml kind'))

    featd = rename_key(featd, nfeats_ml[0], 'sig_ml')
    featd, nfeats_ml2 = perform_cs_appops(featd,
                                          feats=['sig_ml'],
                                          folder=g_folder,
                                          windows=[cfg.get('window_appops')],
                                          name=None,
                                          r=g_r)
    featd = rename_key(featd, nfeats_ml2[0], 'sig_ml_appops')

    # Conditioning on univ - just in case
    for col in get_sig_cols(featd):
        featd[col] = featd[col]*featd['univ']

    # for bench and checking that sharpe is 0.0
    featd['sig_one'] = np.ones_like(featd['wgt'])
    #featd, _ = perform_clean_memory(featd)
    g_r.save()
    return featd


def bktest(featd):
    defargs = {'folder': g_folder, 'name': None, 'r': g_r}

    # Actual backtest
    stats = perform_bktest(featd, **defargs)

    pnlcols = []
    for col in get_sig_cols(featd):
        featd[f'pnl_{col}'] = featd[col]*featd['forward_fh1']
        pnlcols += [f'pnl_{col}']

    # Common Bucketplots
    featd, nfeats = perform_bucketplot(featd,
                                       xcols=['wgt'],
                                       ycols=pnlcols,
                                       n_buckets=8,
                                       freq=int(60*24*1),  # freq quantile updates
                                       **defargs)


def dump_extract(featd):
    """Dumping the data for manual checks"""
    logger.info('Dumping the data for manual checks')

    icols = ['dtsi', 'dscode_str', 'close', 'sig_zs_100', 'sig_mual',
             'univ', 'sret']
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
    #dts = pd.date_range('2022-01-01', '2025-05-01', freq='2MS')
    dts = pd.date_range('2022-01-01', '2025-05-01', freq='1MS')
    for i in range(len(dts)-1):
        start_dt = dts[i].strftime('%Y-%m-%d')
        end_dt = dts[i+1].strftime('%Y-%m-%d')
        featd = main_features(start_date=start_dt, end_date=end_dt)
        # return featd
        # dump_extract(featd)
        # saving down
        # save_signal(featd, name=f'kmeans_signal_{start_dt}_{end_dt}')
        bktest(featd=featd)


# ipython -i -m crptmidfreq.res.mr_cluster_v1.mr_v1
if __name__ == '__main__':
    featd = main()
