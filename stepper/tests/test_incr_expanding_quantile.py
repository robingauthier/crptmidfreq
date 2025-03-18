import pandas as pd

from crptmidfreq.stepper.incr_expanding_quantile import *

# pytest ./stepper/tests/test_incr_quantile_expanding.py --pdb --maxfail=1


###############################################################################
# 2) Test code and utility functions (similar to your EWM tests).
###############################################################################

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
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Choose a quantile
    q = 0.9

    # Create and run QuantileStepper on data
    qstep = QuantileStepper(folder='test_data', name='test_quantile', qs=[q])
    # For an expanding quantile, we feed the entire data once, 
    # but in real streaming you'd feed it in chunks or row by row.
    quant_est = qstep.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'quant_est': quant_est[:,0],
    })
    
    # Calculate actual expanding quantile in pandas for each group.
    # pandas does not have a built-in "groupby expanding quantile" that is 
    # super fast, but we can do an expanding apply:
    # We'll do it code by code, building the same "expanding" approach.
    df['quant_true'] = (
        df.groupby('dscode')['serie'].transform(
          lambda x: x.expanding().quantile(q))
    )

    # Compare results: we can check correlation or compute an average error
    mae = (df['quant_true'] - df['quant_est']).abs().mean()

    print(f"Correlation: MAE: {mae:.4f}")
    # Because T-Digest is approximate, perfect correlation > 0.9 or so is typical
    assert mae < 0.1, f"Expected mae < 0.2, got {mae}"
    # The approximation error might cause differences. Adjust threshold as needed.


def test_save_load():
    """Test save and load functionality for the streaming quantile."""
    n_samples = 1000
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    q = 0.9
    qstep = QuantileStepper(folder='test_data', name='test_quantile', qs=[q])
    # Update with the data
    qstep.update(dt, dscode, serie)
    qstep.save()

    # Load
    qstep_loaded = QuantileStepper.load('test_data', 'test_quantile')
    # Compare q
    assert qstep.qs[0] == qstep_loaded.qs[0]

    # Compare that the same codes exist
    assert qstep.tdigest_map.keys() == qstep_loaded.tdigest_map.keys()


def test_save_load_result():
    """
    Show partial updates (streaming in two chunks), saving, loading, continuing, 
    and comparing with a full expand-based reference.
    """
    n_samples = 1000
    half = n_samples // 2
    n_codes = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)
    q = 0.8

    # 1) Create instance, update with half the data, save
    qstep = QuantileStepper(folder='test_data', name='test_quantile', qs=[q])
    part1 = qstep.update(dt[:half], dscode[:half], serie[:half])
    qstep.save()

    # 2) Load, update with the second half
    qstep_loaded = QuantileStepper.load('test_data', 'test_quantile')
    part2 = qstep_loaded.update(dt[half:], dscode[half:], serie[half:])

    # Combined result
    quant_est = np.concatenate([part1, part2])

    # Compare to a Pandas expanding quantile
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'quant_est': quant_est[:,0],
    })
    
    # Calculate actual expanding quantile in pandas for each group.
    # pandas does not have a built-in "groupby expanding quantile" that is 
    # super fast, but we can do an expanding apply:
    # We'll do it code by code, building the same "expanding" approach.
    df['quant_true'] = (
        df.groupby('dscode')['serie'].transform(
          lambda x: x.expanding().quantile(q))
    )

    # Compare results: we can check correlation or compute an average error
    mae = (df['quant_true'] - df['quant_est']).abs().mean()

    print(f"Correlation:  MAE: {mae:.4f}")
    # Because T-Digest is approximate, perfect correlation > 0.9 or so is typical
    assert mae < 0.1, f"Expected mae < 0.2, got {mae}"
    # The approximation error might cause differences. Adjust threshold as needed.
