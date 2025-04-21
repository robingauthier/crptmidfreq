import numpy as np
import pandas as pd

from crptmidfreq.stepperc.incr_diff import DiffStepper

# pytest ./crptmidfreq/stepperc/tests/test_incr_diff.py --pdb --maxfail=1

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
    window = 1
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run DiffStepper on first half
    diff = DiffStepper(folder='test_data', name='test_diff', window=window)
    seriec = diff.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,
    })

    # Calculate pandas difference
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.diff()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"

    return True
