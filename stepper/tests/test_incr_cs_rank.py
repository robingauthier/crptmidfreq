import pandas as pd

from crptmidfreq.stepper.incr_cs_rank import *

# pytest ./crptmidfreq/stepper/tests/test_incr_cs_rank.py --pdb --maxfail=1



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
    dt = np.sort(np.random.choice(dt0,size=n_samples,replace=True))
    return dt, dscode, serie

def generate_data_hardcoded():
    # Four timestamps, each repeated 5 times (total 20 samples)
    dt = ([1] * 5 +
          [2] * 5 +
          [3] * 5 +
          [4] * 5)
    
    # For simplicity, assign codes that vary within each timestamp group.
    # For example, use codes 1 through 5 for each group.
    dscode = [1, 2, 3, 4, 5] * 4
    
    # Hardcode series values (e.g. some arbitrary float values)
    serie = [10.0, 20.0, 15.0, 30.0, 25.0,
             11.0, 21.0, 16.0, 31.0, 26.0,
             12.0, 22.0, 17.0, 32.0, 27.0,
             13.0, 23.0, 18.0, 33.0, 28.0]
    
    return np.array(dt,dtype=np.int64), np.array(dscode,dtype=np.int64), np.array(serie,dtype=np.float64)

def test_against_pandas():
    import pandas as pd

    # Generate test data
    n_samples = 1000
    n_codes = 10

    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    ewm = csRankStepper(folder='test_data', name='test_ewm')
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
        lambda x: (x.rank(method='dense')-(x.shape[0]+1)/2)/x.shape[0]
    )

    check_ts=df['dt'].value_counts().index[0]
    print(df[df['dt']==check_ts])
    print(df.tail())

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
    

def test_against_pandas2():
    import pandas as pd

    # Generate test data

    dt, dscode, serie = generate_data_hardcoded()

    # Create and run EwmStepper on first half
    ewm = csRankStepper(folder='test_data', name='test_ewm')
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
        lambda x: (x.rank(method='dense')-(x.shape[0]+1)/2)/x.shape[0]
    )

    check_ts=df['dt'].value_counts().index[0]
    print(df[df['dt']==check_ts])
    print(df.tail())

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
    

def test_save_load_result():
    # Generate test data
    n_samples = 1000
    half = int(n_samples / 2)
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create EwmStepper instance
    window = 5
    ewm = csRankStepper(folder='test_data2', name='test_ewm')
    # Update with full data
    serie1 = ewm.update(dt[:half], dscode[:half], serie[:half])
    ewm.save()

    # Load saved state
    ewm_loaded = csRankStepper(folder='test_data2', name='test_ewm')
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
        lambda x: (x.rank(method='dense')-(x.shape[0]+1)/2)/x.shape[0]
    )
    
    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"
    
