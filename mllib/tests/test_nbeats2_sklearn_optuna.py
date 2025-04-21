import os
import pandas as pd
import lightgbm as lgb  # Super important otherwise crashes python
import numpy as np

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.mllib.nbeats2_sklearn_optuna import NBeatsNetOptuna
from crptmidfreq.mllib.tests.test_feedforward import (create_synthetic_data,
                                                      save_some_parquet_files)
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.utils.common import to_csv

np.random.seed(42)

# pytest ./crptmidfreq/mllib/tests/test_nbeats2_sklearn_optuna.py --pdb --maxfail=1

g_folder = os.path.join(get_feature_folder(), 'test_ml')+'/'


def generate_seasonal_data(n_samples, n_codes, season_period=10):
    """
    Generate synthetic data with strong seasonality of a given period.
    Args:
        n_samples (int): Number of time steps in the time series.
        n_codes (int): Number of different codes (e.g., for multiple time series).
        season_period (int): The period of the seasonal cycle (default 10).
    Returns:
        dt (numpy array): Dates of the time series.
        dscode (numpy array): Categorical codes for each time series.
        serie (numpy array): The generated seasonal data with noise.
    """
    np.random.seed(42)

    # Generate dates
    dt = pd.date_range('2020-01-01', periods=n_samples, freq='D').to_numpy()

    # Generate codes for different time series
    dscode = np.repeat(np.arange(n_codes), n_samples // n_codes)

    # Create strong seasonality with noise
    time = np.arange(n_samples)
    seasonality = np.cos(2 * np.pi * time / season_period)  # Seasonal pattern
    trend = 0.1 * time  # Gradual upward trend
    noise = np.random.normal(scale=2.0, size=n_samples)  # Random noise

    # Combine the components: seasonal effect + trend + noise
    serie = 100 + 5 * seasonality + trend   # Base level 10 and amplitude 5 for seasonality

    return dt, dscode, serie, noise


def test_train():
    # 1) Create our streaming dataset

    n_samples = 1000
    n_codes = 1  # Single series for simplicity, can be expanded
    season_period = 10  # Strong seasonality with period 10
    dt, dscode, serie, noise = generate_seasonal_data(n_samples, n_codes, season_period)

    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'noisec': noise,
        'seriec': serie,
        'serie': serie+noise,
    })
    df['forward'] = df['serie'].shift(-1)
    n = df.shape[0]
    dftrain = df.iloc[:n//2].copy()
    dftest = df.iloc[n//2:].copy()

    numfeats = ['serie']
    model = NBeatsNetOptuna(
        input_size=len(numfeats),
    )

    model.fit(dftrain[numfeats], dftrain['forward'])
    print(model)
    ee = model.get_params()['module']()
    total_params = sum(p.numel() for p in ee.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params}")

    # TrendBlock will operate on a batch_size x len(numfeats)
    ypred = model.predict(dftest[numfeats])

    assert np.std(ypred) > 1e-14
    dftest['ypred'] = ypred
    mae = (dftest['ypred'] - dftest['forward']).abs().mean()
    correlation = dftest[['ypred', 'forward']].corr().iloc[0, 1]
    assert correlation > 0.8
    assert mae < 3
    #to_csv(dftest, 'test_nbeats2_sklearn_cos.csv')
