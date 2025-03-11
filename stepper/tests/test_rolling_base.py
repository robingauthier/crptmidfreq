import pandas as pd

from stepper.rolling_base import *
from stepper.tests.test_incr_ewm import generate_data


def test_against_pandas():
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 5
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    sss = RollingStepper(folder='test_data', name='test_ewm', window=window)
    sss.update_memory(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
    })
    df = df.loc[lambda x: x['dscode'] == x['dscode'].min()]
    dscode = df['dscode'].iloc[0]
    assert len(sss.values[dscode]) == window
    assert np.all(np.sort(sss.values[dscode]) ==
                  np.sort(df['serie'].iloc[-window:].values))
