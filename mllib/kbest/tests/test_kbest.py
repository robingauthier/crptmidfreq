import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import torch
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.mllib.iterable_data import ParquetIterableDataset
from crptmidfreq.mllib.feedforward_v1 import FeedForwardNet
from crptmidfreq.mllib.kbest.kbest import perform_kbest
from crptmidfreq.featurelib.lib_v1 import perform_lag
from crptmidfreq.featurelib.lib_v1 import perform_bktest
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.utils.common import rename_key
from crptmidfreq.utils.lazy_dict import LazyDict
from crptmidfreq.utils.common import to_csv
np.random.seed(42)


# pytest ./crptmidfreq/mllib/kbest/tests/test_kbest.py --pdb --maxfail=1 --capture=no


def repeat_sequence_until_n(n, P):
    """
    Returns a NumPy array of length n that repeats [1, 2, 3, ..., P].
    """
    base = np.arange(1, P + 1)
    # Repeat the base sequence enough times, then slice to length n
    return np.tile(base, (n // P + 1))[:n]


def gen_data():
    n = 100_000

    dt = np.arange(n)//10
    start_dt = pd.to_datetime('2024-01-01').value/1e3
    dt = np.int64(start_dt+dt*3600*24*1e6)
    dscode = repeat_sequence_until_n(n, 10)
    clean_folder('test_kbest_dict')
    featd = LazyDict(folder='test_kbest_dict')
    featd['dtsi'] = dt
    featd['dscode'] = dscode
    featd['wgt'] = np.ones_like(dscode)

    # Add some noise
    c = int(n/2)
    true_signal = np.random.normal(0.0, 1, size=n)
    y1 = true_signal + np.random.normal(0.0, 1, size=n)
    y2 = -1*true_signal + np.random.normal(0.0, 1, size=n)
    featd['sigf_1'] = np.concatenate([y1[:c], y2[c:]])
    featd['forward_fh1'] = 0.02*true_signal
    featd, nfeat = perform_lag(featd, ['forward_fh1'], folder='test_kbest')
    featd = rename_key(featd, nfeat[0], 'sret')
    return featd


def test_kbest():
    clean_folder('test_kbest')
    featd = gen_data()
    n = featd['dtsi'].shape[0]
    featd['sigf_2'] = np.random.normal(0.0, 1, size=n)

    featd = perform_kbest(featd,
                          retcol='sret',
                          wgtcol='wgt',
                          window=1000,
                          folder='test_kbest',
                          clip_pnlpct=0.01,  # clip P&L vs gross on stock and daily level
                          name=None,
                          debug=True,
                          sharpe_th=3.0,
                          dd_th=1_000_000.0,
                          rpt_th=2.0,
                          )
    df = pd.DataFrame({k: featd[k] for k in featd.keys()})
    assert abs(df['kbest_sigf_1_sel'].mean()-0.2) < 0.1
    assert (df['sig_kbest'] == 0.0).mean() < 0.2
    assert df['kbest_sigf_1_sel'].iloc[-100:].mean() < -0.9
    assert df['kbest_sigf_1_sel'].iloc[1000:2000].mean() > 0.9
