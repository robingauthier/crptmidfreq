import pandas as pd
from stepper.incr_ewm_detrend import *


def generate_data(n_samples, n_codes):
    """Generate test data"""
    np.random.seed(42)

    # Generate random codes
    dscode = np.random.randint(0, n_codes, n_samples)

    # Generate random series
    serie = np.random.randn(n_samples)

    # Generate increasing datetime
    base = np.datetime64('2024-01-01')
    dt = np.array([base + np.timedelta64(i, 'm') for i in range(n_samples)])

    return dt, dscode, serie


def test_against_pandas():

    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 5
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    ewm = DetrendEwmStepper(folder='test_data', name='test_ewm', window=window)
    seriec=ewm.update(dt, dscode, serie)


    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec':seriec,

    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.ewm(halflife=window, min_periods=1, adjust=False).mean()
    )
    df['serier2']=df['serie']-df['serier']

    # Compare results using correlation
    correlation = df['serier2'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"

    return True
