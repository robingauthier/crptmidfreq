import pandas as pd
import os
from res.mr_cluster_v1.old.kmeans_manual_v1 import g_folder
from crptmidfreq.stepper.incr_bktest import BktestStepper
from crptmidfreq.utils.common import to_csv
from crptmidfreq.utils.common import get_analysis_folder


# more codes in order to investigate the strategy

def dump_dailypnl():
    cls_bk = BktestStepper \
        .load(folder=g_folder, name=f"None_bktest")
    dailypnl = cls_bk.dailypnl
    dailypnl['daily_dt'] = pd.to_datetime(dailypnl['daily_dt']*1e3)
    dailypnl2 = cls_bk.compute_daily_stats()

    to_csv(dailypnl2, 'dailypnl')


def dump_extract(featd):
    """Dumping the data for manual checks"""
    logger.info('Dumping the data for manual checks')

    icols = ['dtsi', 'dscode_str', 'close', 'sig_mual', 'univ', 'kmeans_cat', 'turnover_excess']
    df = pd.DataFrame({k: v for k, v in featd.items() if k in icols})
    df['dtsi'] = pd.to_datetime(df['dtsi']*1e3)

    df_cs_loc = df[df['dtsi'] == df['dtsi'].max()]
    to_csv(df_cs_loc, 'df_cs_loc')

    df_ts_loc = df[df['dscode_str'] == 'BCHUSDT']
    to_csv(df_ts_loc, 'df_ts_loc')

    df_g = df.assign(cnt=1)\
        .groupby('dtsi')\
        .agg({'sig_mual': ('mean', 'std'), 'univ': 'sum', 'cnt': 'sum'})
    to_csv(df_g, 'df_univ_cnt')

    # all the columns now
    featd_loc = filter_dict_to_dscode(featd, dscode_str='BCHUSDT')
    df_loc = pd.DataFrame(featd_loc)
    df_loc['dtsi'] = pd.to_datetime(df_loc['dtsi']*1e3)
    to_csv(df_loc, 'df_ts_loc_allcols')

    dtsi = np.max(featd['dtsi'])
    featd_loc = filter_dict_to_dts(featd, dtsi=dtsi)
    df_loc = pd.DataFrame(featd_loc)
    df_loc['dtsi'] = pd.to_datetime(df_loc['dtsi']*1e3)
    to_csv(df_loc, 'df_cs_loc_allcols')

# ipython -i -m crptmidfreq.res.mr_cluster_v1.investigate_v1
if __name__ == '__main__':
    df = pd.read_parquet(os.path.join(
        get_analysis_folder(),
        'kmeans_manual_v1_features_2022-02-01_2022-03-20.pq'))
    to_csv(df.sample(10000), 'debug')
