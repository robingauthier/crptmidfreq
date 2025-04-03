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
from crptmidfreq.utils.common import merge_dicts
from crptmidfreq.utils.common import to_csv
from crptmidfreq.mllib.lgbm_lin_v1 import gen_lgbm_lin_v1
from crptmidfreq.mllib.feedforward_v1 import gen_feed_forward
from crptmidfreq.mllib.boosting_torch import gen_boosting_torch
from crptmidfreq.mllib.train_lgbm import gen_lgbm_lin_params
from crptmidfreq.mllib.linear_torch import gen_linear_torch
from crptmidfreq.utils.univ import hardcoded_universe_1
from crptmidfreq.mllib.kbest.kbest import perform_kbest

# This will give you intuitions as to what kind of strategies should work

logger = get_logger()

g_folder = 'res_kmeans_naive_v1'
g_r = StepperRegistry()


def main_features(start_date='2025-03-01', end_date='2026-01-01', icfg={}):
    logger.info(f'mr_cluster start_date={start_date} end_date={end_date}')
    # all the hyper parameters
    unit_day = 60*24

    dcfg = dict(
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
        hardcoded_universe=True,  # TODO: roll back

        # applyops
        window_appops=1000,

        nb_fetures=1,  # 1,2  1 is minimum amount of features

        # gap features
        gaps_lag=1,
        gaps_window=[unit_day, 5*unit_day, 15*unit_day],
        gaps_th=[2.0, 4.0],
        gaps_ewm_windows=[10, 100],

        model_lookback=unit_day*400,  # I get into RAM issues otherwis
        model_minlookback=unit_day*20,
        model_fitfreq=unit_day*100,

        # ml_kind='mllgbm',
        ml_kind='kbest',

    )
    if dcfg['hardcoded_universe']:
        dcfg['kmeans_k'] = 2
        dcfg['svd_k'] = 2
    cfg = merge_dicts(icfg, dcfg)
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
    elif kind == 'mkt':
        featd['sret'] = featd['tret_xmkt']  # it is clipped yes
    elif kind == 'tret_turnover':
        featd['sret'] = featd['tret']*featd['turnover']
    elif kind == 'tret_turnover_cs':
        featd['sret'] = featd['tret_xmkt']*featd['turnover']/featd['wgt']
    else:
        featd['sret'] = featd['tret']

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
                          feats=['sret'],
                          **defargs)

        # P&L features
        featd = pnl_feats(featd,
                          incol='mual',
                          fretcol='sret',
                          **defargs)

        # volatility features
        featd = volatility_feats(featd,
                                 feats=['sret'],
                                 **defargs)

    # gap features
    featd = gap_feats(featd,
                      incol='sret',
                      folder=g_folder,
                      name=None,
                      r=g_reg,
                      cfg=icfg)

    if cfg.get('ml_kind') == 'no_ml':
        # Converting sigf to sig
        for col in get_sigf_cols(featd):
            ncol = col.replace('sigf_', 'sig_')
            featd = rename_key(featd, col, ncol)
    elif cfg.get('ml_kind') == 'kbest':
        featd = perform_kbest(featd,
                              retcol='tret_xmkt',
                              wgtcol='wgt',
                              window=cfg.get('model_lookback'),
                              folder=g_folder,
                              clip_pnlpct=0.05,  # clip P&L vs gross on stock and daily level
                              name="None",
                              r=g_r)
    elif cfg.get('ml_kind') == 'ml':
        featd = filter_dict_to_univ(featd)
        featd, nfeats_ml = perform_model(
            featd,
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
        featd = rename_key(featd, nfeats_ml[0], 'sig_ml')
    elif cfg.get('ml_kind') == 'mlpytorch':
        featd = filter_dict_to_univ(featd)
        featd, nfeats_ml = perform_model_batch(
            featd,
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
            is_torch=True,
            #model_gen=partial(gen_feed_forward, hidden_dim=20),
            #model_gen=partial(gen_boosting_torch, hidden_dim=20),
            with_fit=True,
            r=g_r)
        featd = rename_key(featd, nfeats_ml[0], 'sig_ml')
    elif cfg.get('ml_kind') == 'mllgbm':
        featd = filter_dict_to_univ(featd)
        fcols = [f'forward_fh{win}_clip' for win in cfg.get('windows_forward')]
        for win in [1]+cfg.get('windows_forward'):
            featd, nfeats_ml = perform_model_batch(
                featd,
                feats=get_sigf_cols(featd),
                wgt='wgt',
                ycol=f'forward_fh{win}_clip',
                folder=g_folder,
                name="None",
                lookback=cfg.get('model_lookback'),
                minlookback=cfg.get('model_minlookback'),
                ramlookback=10*unit_day,  # if 1day it is too slow
                batch_size=300,
                epochs=1,
                lr=1e-4,
                weight_decay=1e-2,
                fitfreq=cfg.get('model_fitfreq'),
                gap=1,
                model_gen=gen_lgbm_lin_params,
                is_torch=False,
                with_fit=True,
                r=g_r)
            featd = rename_key(featd, nfeats_ml[0], f'sig_ml_{win}')
    else:
        raise(ValueError('issue ml kind'))

    # now ewms of sigs
    sig_ml_cols = [x for x in featd.keys() if x.startswith('sig_ml')]
    featd, nfeats_ml_ewm = perform_ewm(featd,
                                       feats=sig_ml_cols,
                                       folder=g_folder,
                                       windows=[2, 5, 10],
                                       name=None,
                                       r=g_r)
    featd, nfeats_ml2 = perform_cs_appops(featd,
                                          feats=sig_ml_cols+nfeats_ml_ewm,
                                          folder=g_folder,
                                          windows=[cfg.get('window_appops')],
                                          name=None,
                                          r=g_r)
    #featd = rename_key(featd, nfeats_ml2[0], 'sig_ml_appops')

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
    featd, nfeats = perform_bucketplot(featd,
                                       xcols=['sigf_timeofday'],
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


def main(icfg={}):
    clean_folder(g_folder)
    #dts = pd.date_range('2022-01-01', '2025-05-01', freq='2MS')
    #dts = pd.date_range('2022-01-01', '2025-05-01', freq='1MS')
    dts = pd.date_range('2022-01-01', '2025-05-01', freq='20D')
    for i in range(len(dts)-1):
        start_dt = dts[i].strftime('%Y-%m-%d')
        end_dt = dts[i+1].strftime('%Y-%m-%d')
        featd = main_features(start_date=start_dt, end_date=end_dt, icfg=icfg)
        # return featd
        # dump_extract(featd)
        # saving down
        # save_signal(featd, name=f'kmeans_signal_{start_dt}_{end_dt}')
        bktest(featd=featd)


# ipython -i -m crptmidfreq.res.mr_cluster_v1.mr_v1
if __name__ == '__main__':
    if False:
        featd = main(icfg={
            'forward_xmkt': False,
            'kmeans_or_svd_or_naive': 'tret',
            'hardcoded_universe': True,
            'nb_fetures': 1,
        })
    if False:
        # this one works
        featd = main(icfg={
            'forward_xmkt': True,
            'kmeans_or_svd_or_naive': 'mkt',
            'hardcoded_universe': False,
            'nb_fetures': 1,
        })
    if True:
        # complex
        featd = main(icfg={
            'forward_xmkt': False,
            'kmeans_or_svd_or_naive': 'mkt',
            'hardcoded_universe': True,
            'nb_fetures': 2,
        })
    if False:
        # this one below is good
        featd = main(icfg={
            'forward_xmkt': True,
            'kmeans_or_svd_or_naive': 'tret_turnover_cs',
            'hardcoded_universe': True,
            'nb_fetures': 1,
        })
