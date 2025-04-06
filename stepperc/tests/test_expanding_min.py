import pandas as pd

from crptmidfreq.stepperc.incr_expanding_min import *
from crptmidfreq.stepperc.tests.test_incr_ewm import generate_data

# pytest ./crptmidfreq/stepperc/tests/test_expanding_min.py --pdb --maxfail=1


def test_against_pandas():
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 5
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    sss = MinStepper(folder='test_data', name='test_ewm')
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
        lambda x: x.expanding().min()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.98, f"Expected correlation >0.9, got {correlation}"
    assert (df['serier'] - df['seriec']).abs().max() < 1e-5
