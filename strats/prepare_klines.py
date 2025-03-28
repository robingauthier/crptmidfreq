import pandas as pd
import os
import duckdb
from crptmidfreq.utils.common import rename_key
from crptmidfreq.config_loc import get_data_db_folder
from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import merge_dicts


logger = get_logger()


def prepare_klines(start_date='2025-03-01',
                   end_date='2026-01-01',
                   tokens='all',  # tokens=['BTCUSDT', 'ETHUSDT']
                   folder=None,
                   name=None,
                   r=None,  # stepper registry
                   cfg={}
                   ):
    """
    Reads the klines from the SQL database
    - computes tret and tret_qtlclip
    - computes wgt
    """
    assert not folder is None
    dcfg = dict(
        window_volume_wgt=60*24*30,
    )
    cfg = merge_dicts(cfg, dcfg, name='kmeans_sret')

    logger.info(f'prepare_klines start_date={start_date} end_date={end_date}')

    if tokens == 'all':
        token_cond = ''
    else:
        list_tokens_str = '\',\''.join(tokens)
        list_tokens_str = '(\''+list_tokens_str+'\')'
        token_cond = f' AND dscode IN {list_tokens_str}'

    logger.info('Reading data from DuckDB')
    con = duckdb.connect(os.path.join(get_data_db_folder(), "my_database.db"), read_only=True)
    df = con.execute(f'''SELECT close_time,dscode,close,volume,taker_buy_volume 
                   FROM klines
                   WHERE CAST(close_time AS DATE)>='{start_date}'
                   {token_cond}
                   AND CAST(close_time AS DATE)<'{end_date}';
                   ''').df()

    # we need to convert dscode to an integer
    df['dscode_str'] = df['dscode'].copy()
    cat = pd.Categorical(df['dscode_str'])
    df['dscode'] = cat.codes
    df['dscode'] = df['dscode'].astype('int64')  # defaults to int16
    df['dtsi'] = df['close_time'].astype('int64')
    df.sort_values('close_time', ascending=True, inplace=True)

    # working on numpy now
    featd = {col: df[col].values for col in df.columns}

    # Casing to float64 -- otherwise we have issues later
    featd, _ = perform_cast_float64(featd,
                                    feats=['close', 'volume', 'taker_buy_volume'],
                                    folder=folder,
                                    name=name,
                                    r=r)

    # turnover = volume*close  should be in USDT
    featd['turnover'] = featd['volume']*featd['close']

    # adding distance since IPO  :: cnt_exists
    featd, nfeats = perform_cnt_exists(featd=featd,
                                       feats=[],
                                       folder=folder,
                                       name=name,
                                       r=r)
    featd = rename_key(featd, 'cnt_exists', 'sigf_ipocnt')

    # adding the time of the day -- I confirm it works
    one_day_unit = int(3600*24*1e6)
    featd['sigf_timeofday'] = np.mod(featd['dtsi'], one_day_unit)
    featd['sigf_timeofday'] = featd['sigf_timeofday']/one_day_unit*24

    # adding returns
    featd, nfeats = perform_diff(featd=featd,
                                 feats=['close'],
                                 windows=[1],
                                 folder=folder,
                                 name=name,
                                 r=r)
    featd['tret'] = np.divide(
        featd[nfeats[0]],
        featd['close'],
        out=np.zeros_like(featd['close']),
        where=~np.isclose(featd['close'], np.zeros_like(featd['close'])))

    # Clip tret :: called tret_clipqtl
    featd, nfeats = perform_clip_quantile_global(featd,
                                                 feats=['tret'],
                                                 low_clip=0.05,
                                                 high_clip=0.95,
                                                 folder=folder,
                                                 name=name,
                                                 r=r)

    
    # adding the weight ewm(volume) clipped to X% quantile
    featd, nfeats_wgt1 = perform_ewm(featd=featd,
                                     feats=['turnover'],
                                     windows=[cfg.get('window_volume_wgt')],
                                     folder=folder,
                                     name=name,
                                     r=r)

    featd, nfeats_wgt2 = perform_quantile_global(featd,
                                                 feats=nfeats_wgt1,
                                                 qs=[0.8],
                                                 folder=folder,
                                                 name=name,
                                                 r=r)

    featd['wgt'] = np.where(featd[nfeats_wgt1[0]] < featd[nfeats_wgt2[0]],
                            featd[nfeats_wgt1[0]],
                            featd[nfeats_wgt2[0]])
    featd['wgt'] = np.nan_to_num(featd['wgt'])
    featd['sigf_wgt'] = featd['wgt']
    return featd
