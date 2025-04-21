import pandas as pd

from crptmidfreq.stepper.incr_cs_mean import *

# pytest ./crptmidfreq/stepper/tests/test_incr_cs_mean.py --pdb --maxfail=1


def generate_data(n_samples, n_codes):
    """Generate test data"""
    np.random.seed(42)

    # Generate random codes
    dscode = np.random.randint(0, n_codes, n_samples)

    # Generate random series
    serie = np.random.randn(n_samples)

    # Generate increasing datetime
    base = np.datetime64('2024-01-01')
    dt0 = np.array([base + np.timedelta64(i, 'm') for i in range(n_samples)])
    dt = np.sort(np.random.choice(dt0, size=n_samples, replace=True))
    return dt, dscode, serie


def test_against_pandas():
    import pandas as pd

    # Generate test data
    n_samples = 100000
    n_codes = 10

    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    ewm = csMeanStepper(folder='test_data', name='test_ewm')
    seriec = ewm.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,

    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dt')['serie'].transform(
        lambda x: x.mean()
    )

    check_ts = df['dt'].value_counts().index[0]
    print(df[df['dt'] == check_ts])
    print(df.tail())

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
    mae = (df['serier']-df['seriec']).abs().max()
    assert mae < 1e-6


def test_against_pandas_withby():
    import pandas as pd

    # Generate test data
    n_samples = 100000
    n_codes = 10

    dt, dscode, serie = generate_data(n_samples, n_codes)
    by = dscode = np.random.randint(0, 2, n_samples)

    # Create and run EwmStepper on first half
    ewm = csMeanStepper(folder='test_data', name='test_ewm')
    seriec = ewm.update(dt, dscode, serie, by=by)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'by': by,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,

    })

    # Calculate pandas EWM
    df['serier'] = df.groupby(['dt', 'by'])['serie'].transform(
        lambda x: x.mean()
    )

    check_ts = df['dt'].value_counts().index[0]
    print(df[df['dt'] == check_ts])
    print(df.tail())

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
    mae = (df['serier']-df['seriec']).abs().max()
    assert mae < 1e-6


def test_against_pandas_zero():
    pass
    # Generate test data

    dt = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    dscode = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)
    serie = np.array([-1, 1, 0, 0, 0, 0, -6, 3, 3], dtype=np.float64)
    wgt = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    # Create and run EwmStepper on first half
    ewm = csMeanStepper(folder='test_data', name='test_ewm')
    seriec = ewm.update(dt, dscode, serie, wgt=wgt)

    # Create pandas DataFrame for comparison
    assert np.max(np.abs(seriec)) < 1e-6


def test_against_pandas_zero_wgt():
    pass

    dt = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    dscode = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)
    serie = np.array([-1, 1, 0, 0, 0, 0, -6, 3, 3], dtype=np.float64)
    wgt = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Create and run EwmStepper on first half
    ewm = csMeanStepper(folder='test_data2', name='test_ewm')
    seriec = ewm.update(dt, dscode, serie, wgt=wgt)

    # Create pandas DataFrame for comparison
    assert np.max(np.abs(seriec)) < 1e-6


def test_save_load_result():
    # Generate test data
    n_samples = 1000
    half = int(n_samples / 2)
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create EwmStepper instance
    window = 5
    ewm = csMeanStepper(folder='test_data2', name='test_ewm')
    # Update with full data
    serie1 = ewm.update(dt[:half], dscode[:half], serie[:half])
    ewm.save()

    # Load saved state
    ewm_loaded = csMeanStepper(folder='test_data2', name='test_ewm')
    serie2 = ewm_loaded.update(dt[half:], dscode[half:], serie[half:])

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': np.concatenate([serie1, serie2]),
    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dt')['serie'].transform(
        lambda x: x.mean()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
    mae = (df['serier']-df['seriec']).abs().max()
    assert mae < 1e-6
