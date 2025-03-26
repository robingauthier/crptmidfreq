import numpy as np
import pandas as pd
from crptmidfreq.stepper.incr_expanding_bucketxy_fast import BucketXYStepper
from crptmidfreq.utils.common import clean_folder


# pytest ./crptmidfreq/stepper/tests/test_incr_expanding_bucketxy.py --pdb --maxfail=1


def test_bucketplot():
    # Clean up or create a test folder
    folder = "test_data"
    clean_folder(folder)
    # 1) Create stepper with 8 buckets
    n_buckets = 8
    stepper = BucketXYStepper(folder=folder, name="test_bucket", n_buckets=n_buckets, freq=1)

    # 2) Generate random data
    np.random.seed(42)
    n_samples = 1000
    dt_values = np.arange(n_samples)
    dscode_values = np.ones(n_samples, dtype=np.int64)
    x_values = np.random.randn(n_samples)
    y_values = 2.0 * x_values + 0.5 * np.random.randn(n_samples)

    # 3) Update: dt=None, dscode=None for now
    results_mean, results_std = stepper.update(dt=dt_values, dscode=dscode_values,
                                               x_values=x_values, y_values=y_values)
    df_buckets = pd.DataFrame({'x': range(n_buckets),
                              'y': results_mean[-1],
                               'y_std': results_std[-1]})

    # checking manually
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    df['xbin'] = pd.qcut(df['x'], n_buckets)
    df_r = df.groupby('xbin').agg({'y': 'mean'})
    df_buckets['y_c'] = df_r['y'].values
    print(df_buckets)
    mae = (df_buckets['y_c']-df_buckets['y']).abs().mean()
    assert mae < 0.5
    corr = df_buckets[['y_c', 'y']].corr().iloc[0, 1]
    assert corr > 0.9


def test_bucketplot_load():
    # Clean up or create a test folder
    folder = "test_data"
    clean_folder(folder)
    # 1) Create stepper with 8 buckets
    n_buckets = 8

    # 2) Generate random data
    np.random.seed(42)
    n_samples = 1000
    dt_values = np.arange(n_samples)
    dscode_values = np.ones(n_samples, dtype=np.int64)
    x_values = np.random.randn(n_samples)
    y_values = 2.0 * x_values + 0.5 * np.random.randn(n_samples)

    c = 500
    # 3) Update: dt=None, dscode=None for now
    stepper = BucketXYStepper.load(folder=folder, name="test_bucket", n_buckets=n_buckets, freq=1)
    results_mean, results_std = stepper.update(dt=dt_values[:c],
                                               dscode=dscode_values[:c],
                                               x_values=x_values[:c],
                                               y_values=y_values[:c])
    stepper.save()
    stepper2 = BucketXYStepper.load(folder=folder, name="test_bucket", n_buckets=n_buckets, freq=1)
    results_mean, results_std = stepper2.update(dt=dt_values[c:],
                                                dscode=dscode_values[c:],
                                                x_values=x_values[c:],
                                                y_values=y_values[c:])

    df_buckets = pd.DataFrame({'x': range(n_buckets),
                              'y': results_mean[-1],
                               'y_std': results_std[-1]})

    # checking manually
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    df['xbin'] = pd.qcut(df['x'], n_buckets)
    df_r = df.groupby('xbin').agg({'y': 'mean'})
    df_buckets['y_c'] = df_r['y'].values
    print(df_buckets)
    mae = (df_buckets['y_c']-df_buckets['y']).abs().mean()
    assert mae < 0.5
    corr = df_buckets[['y_c', 'y']].corr().iloc[0, 1]
    assert corr > 0.9
