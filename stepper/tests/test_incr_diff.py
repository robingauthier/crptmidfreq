import pandas as pd
from stepper.incr_diff import *

# pytest ./feature/tests/test_incr_diff.py --pdb --maxfail=1

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


def test_save_load():
    """Test save and load functionality"""
    n_samples = 1000
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)
    window = 1

    # Create and update original instance
    diff = DiffStepper(folder='test_data', name='test_diff', window=window)
    diff.update(dt, dscode, serie)
    diff.save()

    # Load saved state
    diff_loaded = DiffStepper.load('test_data', 'test_diff')

    # Compare states
    assert diff.window == diff_loaded.window
    for code in diff.diff_values.keys():
        assert diff.last_timestamps[code] == diff_loaded.last_timestamps[code]


def test_save_load_result():

    # Generate test data
    n_samples = 1000
    half = int(n_samples / 2)
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create DiffStepper instance
    window = 1
    diff = DiffStepper(folder='test_data', name='test_diff', window=window)
    # Update with full data
    serie1 = diff.update(dt[:half], dscode[:half], serie[:half])
    diff.save()

    # Load saved state
    diff_loaded = DiffStepper.load('test_data', 'test_diff')
    assert diff.window == diff_loaded.window
    for code in diff.last_timestamps.keys():
        assert diff.last_timestamps[code] == diff_loaded.last_timestamps[code]
    serie2 = diff_loaded.update(dt[half:], dscode[half:], serie[half:])



    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': np.concatenate([serie1, serie2]),
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

