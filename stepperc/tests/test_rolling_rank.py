from crptmidfreq.stepper.rolling_rank import *

# pytest ./crptmidfreq/stepper/tests/test_rolling_rank.py --pdb --maxfail=1


def generate_data(n_samples, n_codes):
    """Generate test data"""
    np.random.seed(42)

    # Generate random codes
    dscode = np.random.randint(0, n_codes, n_samples)

    # Generate random series
    serie = np.random.randn(n_samples)

    # Generate increasing datetime
    base = np.datetime64('2024-01-01')
    dt = np.array([base + np.timedelta64(i, 'm') for i in range(n_samples)], dtype='datetime64')

    return dt, dscode, serie


def test_against_pandas():
    import pandas as pd
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    ewm = RollingRankStepper(folder='test_data', name='test_ewm', window=window)
    seriec = ewm.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,
    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.rolling(window=window).rank()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
