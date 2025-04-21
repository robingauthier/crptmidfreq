import pandas as pd

from crptmidfreq.stepper.incr_theta import *
from crptmidfreq.stepper.tests.test_incr_ewm import generate_data

# pytest ./crptmidfreq/stepper/tests/test_theta.py --pdb --maxfail=1


def test_theta1():
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 100
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    sss = ThetaStepper(folder='test_data', name='test_ewm', window=window,theta=1.0)
    seriec = sss.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,
    })

    # Calculate pandas EWM
    df['serier'] = df['serie']

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.98, f"Expected correlation >0.9, got {correlation}"
    assert (df['serier'] - df['seriec']).abs().max() < 1e-5

def test_theta0():
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 100
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    sss = ThetaStepper(folder='test_data', name='test_ewm', window=window,theta=0.0)
    seriec = sss.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,
    })

    # Calculate pandas EWM
    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.ewm(halflife=10*window, min_periods=1).mean()
    )
    
    #from crptmidfreq.utils.common import to_csv
    #to_csv(df,'test_theta0.csv')
    #import pdb;pdb.set_trace()
    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.98, f"Expected correlation >0.9, got {correlation}"
    assert (df['serier'] - df['seriec']).abs().max() < 1e-5