import shutil

import pandas as pd
from features.lib_v1 import *

from crptmidfreq.config_loc import get_data_folder
from crptmidfreq.stepper.tests.test_incr_ewm import generate_data


def test_perform_lag_load():
    # Generate test data
    n_samples = 1000
    n_half = int(n_samples / 2)
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    featd1 = {
        'dtsi': dt[:n_half].astype(np.int64),
        'dscode': dscode[:n_half],
        's': serie[:n_half],
    }
    featd2 = {
        'dtsi': dt[n_half:].astype(np.int64),
        'dscode': dscode[n_half:],
        's': serie[n_half:],
    }
    try:
        shutil.rmtree(get_data_folder() + '/test_data')
    except Exception as e:
        pass
    featd1, f0 = perform_lag(featd1, ['s'], [20], folder='test_data', name='test2')
    featd2, f0 = perform_lag(featd2, ['s'], [20], folder='test_data', name='test2')

    featd3 = {
        'dtsi': dt.astype(np.int64),
        'dscode': dscode,
        's': serie,
        's_ewm5': np.concatenate([featd1[f0[0]], featd2[f0[0]]]),
    }
    df = pd.DataFrame(featd3)

    df['serier'] = df.groupby('dscode')['s'].transform(
        lambda x: x.shift(20)
    )
    df['dst'] = (df['serier'] - df['s_ewm5']).abs()
    # Compare results using correlation
    correlation = df['serier'].corr(df['s_ewm5'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.95, f"Expected correlation >0.9, got {correlation}"
    assert df['dst'].max() < 1e-4
