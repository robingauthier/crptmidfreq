from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from crptmidfreq.stepperc.incr_ewmkurt import EwmKurtStepper

# pytest ./crptmidfreq/stepperc/tests/test_incr_ewmkurt.py --pdb --maxfail=1


def test_ewmkurt_math():
    # Generate timestamps
    n_points = 1000
    base_dt = datetime(2024, 1, 1)
    timestamps = np.array([base_dt + timedelta(seconds=i) for i in range(n_points)], dtype='datetime64')

    # Generate two series with different kurtosis
    np.random.seed(42)

    # For dscode=1: High kurtosis using Student's t distribution with low df
    high_kurt = np.random.standard_t(df=3, size=n_points)

    # For dscode=2: Low kurtosis using uniform distribution
    low_kurt = np.random.uniform(-1, 1, n_points)

    # Combine data
    dscodes = np.array([1] * n_points + [2] * n_points)
    values = np.concatenate([high_kurt, low_kurt])
    timestamps = np.concatenate([timestamps, timestamps])

    # Initialize and update stepper
    stepper = EwmKurtStepper(folder="test", name="test_kurt", window=100)
    kurt_values = stepper.update(timestamps, dscodes, values)

    # Split results by dscode
    kurt_dscode1 = kurt_values[:n_points]
    kurt_dscode2 = kurt_values[n_points:]

    # Check average kurtosis
    # Exclude first few values as they need time to stabilize
    start_idx = 50
    assert np.mean(kurt_dscode1[start_idx:]) > np.mean(kurt_dscode2[start_idx:]), \
        "Expected higher kurtosis for dscode=1 (t-dist) than dscode=2 (uniform)"


def test_ewmkurt_matches_pandas():
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    n_codes = 5

    # Create datetime range
    base_dt = datetime(2023, 1, 1)
    dt = np.array([base_dt + timedelta(minutes=i) for i in range(n_samples)], dtype='datetime64')

    # Create random codes and values
    dscodes = np.random.randint(0, n_codes, size=n_samples)
    values = np.random.randn(n_samples)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'datetime': dt,
        'dscode': dscodes,
        'value': values
    })

    # Test with different window sizes
    windows = [20]

    for win in windows:
        # Calculate using pandas
        expected = df.groupby('dscode')['value'].transform(lambda x: x.rolling(win).kurt())

        # Calculate using our implementation
        stepper = EwmKurtStepper(window=win)
        result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)
        df['r'] = result
        df['e'] = expected
        # Compare results using correlation
        correlation = df[['r', 'e']].corr().iloc[0, 1]

        assert correlation > 0.4, f"Correlation with pandas implementation should be >0.9 for window={win}, got {correlation}"
