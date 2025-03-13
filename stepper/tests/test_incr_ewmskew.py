import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from ..incr_ewmskew import EwmSkewStepper
# pytest ./stepper/tests/test_incr_ewmskew.py --pdb --maxfail=1

def test_ewmskew_math():
    # Generate timestamps
    n_points = 1000
    base_dt = datetime(2024, 1, 1)
    timestamps = np.array([base_dt + timedelta(seconds=i) for i in range(n_points)],dtype='datetime64')
    
    # Generate two series with different skewness
    np.random.seed(42)
    
    # For dscode=1: Positive skew using lognormal distribution
    pos_skew = np.random.lognormal(0, 0.7, n_points)
    
    # For dscode=2: Negative skew using beta distribution
    neg_skew = np.random.beta(5, 2, n_points)
    
    # Combine data
    dscodes = np.array([1] * n_points + [2] * n_points)
    values = np.concatenate([pos_skew, neg_skew])
    timestamps = np.concatenate([timestamps, timestamps])
    
    # Initialize and update stepper
    stepper = EwmSkewStepper(folder="test", name="test_skew", window=100)
    skew_values = stepper.update(timestamps, dscodes, values)
    
    # Split results by dscode
    skew_dscode1 = skew_values[:n_points]
    skew_dscode2 = skew_values[n_points:]
    
    # Check average skewness signs
    # Exclude first few values as they need time to stabilize
    start_idx = 50
    assert np.mean(skew_dscode1[start_idx:]) > 0, "Expected positive skew for dscode=1"
    assert np.mean(skew_dscode2[start_idx:]) < 0, "Expected negative skew for dscode=2"

def test_ewmskew_matches_pandas():
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    n_codes = 5
    
    # Create datetime range
    base_dt = datetime(2023, 1, 1)
    dt = np.array([base_dt + timedelta(minutes=i) for i in range(n_samples)],dtype='datetime64')
    
    # Create random codes and values
    dscodes = np.random.randint(0, n_codes, size=n_samples)
    values = np.random.randn(n_samples)
    
    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'datetime': dt,
        'dscode': dscodes,
        'value': values
    })
    
    # Test with different window sizes
    windows = [5, 10, 20]
    
    for win in windows:
        # Calculate using pandas
        expected = df.groupby('dscode')['value'].transform(lambda x: x.rolling(win).skew())
        
        # Calculate using our implementation
        stepper = EwmSkewStepper(window=win)
        result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)

        # Compare results using correlation
        df['e'] = expected
        df['r'] = result
        correlation =df['r'].corr(df['e'])
        assert correlation > 0.4, f"Correlation with pandas implementation should be >0.9 for window={win}, got {correlation}"

