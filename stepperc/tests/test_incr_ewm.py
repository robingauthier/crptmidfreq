import numpy as np
import pandas as pd

from crptmidfreq.stepperc.incr_ewm import EwmStepper

# pytest ./crptmidfreq/stepperc/tests/test_incr_ewm.py --pdb --maxfail=1

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
    ewm = EwmStepper(folder='test_data', name='test_ewm', window=window)
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
        lambda x: x.ewm(halflife=window, min_periods=1).mean()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"



def test_save_load():
    """Test save and load functionality"""
    n_samples = 1000
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)
    window = 20

    # Create and update original instance
    ewm = EwmStepper(folder='test_data', name='test_ewm', window=window)
    ewm.update(dt, dscode, serie)
    ewm.save()

    # Load saved state
    ewm_loaded = EwmStepper.load('test_data', 'test_ewm')

    # Compare states
    assert ewm.window == ewm_loaded.window
    assert ewm.alpha == ewm_loaded.alpha


def test_save_load_result():
    # Generate test data
    n_samples = 1000
    half = int(n_samples / 2)
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create EwmStepper instance
    window = 5
    ewm = EwmStepper(folder='test_data', name='test_ewm', window=window)
    # Update with full data
    serie1 = ewm.update(dt[:half], dscode[:half], serie[:half])
    ewm.save()

    # Load saved state
    ewm_loaded = EwmStepper.load('test_data', 'test_ewm')
    serie2 = ewm_loaded.update(dt[half:], dscode[half:], serie[half:])

    assert ewm.window == ewm_loaded.window
    assert ewm.alpha == ewm_loaded.alpha

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': np.concatenate([serie1, serie2]),
    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.ewm(halflife=window, min_periods=1, adjust=False).mean()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"


