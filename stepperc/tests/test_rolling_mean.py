import pandas as pd

from crptmidfreq.stepper.rolling_mean import *
from crptmidfreq.stepper.tests.test_incr_ewm import generate_data


def test_against_pandas():
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 5
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    sss = RollingMeanStepper(folder='test_data', name='test_ewm', window=window)
    seriec = sss.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,
    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.98, f"Expected correlation >0.9, got {correlation}"
    assert (df['serier'] - df['seriec']).abs().max() < 1e-5


def test_save_load_result():
    # Generate test data
    n_samples = 1000
    half = int(n_samples / 2)
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create EwmStepper instance
    window = 5
    ewm = RollingMeanStepper(folder='test_data', name='test_sma', window=window)
    # Update with full data
    serie1 = ewm.update(dt[:half], dscode[:half], serie[:half])
    ewm.save()

    # Load saved state
    ewm_loaded = RollingMeanStepper.load('test_data', 'test_sma')
    serie2 = ewm_loaded.update(dt[half:], dscode[half:], serie[half:])

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': np.concatenate([serie1, serie2]),
    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
