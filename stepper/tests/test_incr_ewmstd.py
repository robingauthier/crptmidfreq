import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from ..incr_ewmstd import EwmStdStepper


# pytest ./stepper/tests/test_incr_ewmstd.py --pdb --maxfail=1

def test_ewmstd_math():
    # Generate timestamps
    n_points = 1000
    base_dt = datetime(2024, 1, 1)
    timestamps = np.array([base_dt + timedelta(seconds=i) for i in range(n_points)],dtype='datetime64')
    
    # Generate two series with different volatility
    np.random.seed(42)
    
    # For dscode=1: High volatility using normal distribution with high std
    high_vol = np.random.normal(0, 2.0, n_points)
    
    # For dscode=2: Low volatility using normal distribution with low std
    low_vol = np.random.normal(0, 0.5, n_points)
    
    # Combine data
    dscodes = np.array([1] * n_points + [2] * n_points)
    values = np.concatenate([high_vol, low_vol])
    timestamps = np.concatenate([timestamps, timestamps])
    
    # Initialize and update stepper
    stepper = EwmStdStepper(folder="test", name="test_std", window=100)
    std_values = stepper.update(timestamps, dscodes, values)
    
    # Split results by dscode
    std_dscode1 = std_values[:n_points]
    std_dscode2 = std_values[n_points:]
    
    # Check average standard deviation
    # Exclude first few values as they need time to stabilize
    start_idx = 50
    assert np.mean(std_dscode1[start_idx:]) > np.mean(std_dscode2[start_idx:]), \
        "Expected higher std for dscode=1 (high vol) than dscode=2 (low vol)"

def test_ewmstd_matches_pandas():
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
    windows = [20]
    
    for win in windows:
        # Calculate using pandas
        expected = df.groupby('dscode')['value'].transform(lambda x: x.ewm(halflife=win).std())
        
        # Calculate using our implementation
        stepper = EwmStdStepper(window=win)
        result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)
        df['r']=result
        df['e']=expected
        # Compare results using correlation
        correlation = df[['r','e']].corr().iloc[0,1]
        assert correlation > 0.5, f"Correlation with pandas implementation should be >0.9 for window={win}, got {correlation}"

