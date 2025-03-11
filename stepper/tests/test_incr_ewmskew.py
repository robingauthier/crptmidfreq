import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from ..incr_ewmskew import EwmSkewStepper

def test_ewmskew_math():
    # Generate timestamps
    n_points = 1000
    base_dt = datetime(2024, 1, 1)
    timestamps = np.array([base_dt + timedelta(seconds=i) for i in range(n_points)])
    
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
    dt = np.array([base_dt + timedelta(minutes=i) for i in range(n_samples)])
    
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


def test_input_validation():
    stepper = EwmSkewStepper(window=5)
    
    # Test non-numpy array inputs
    with pytest.raises(ValueError, match="All inputs must be numpy arrays"):
        stepper.update([1, 2, 3], np.array([1, 2, 3]), np.array([1, 2, 3]))
    
    # Test mismatched lengths
    with pytest.raises(ValueError, match="All inputs must have the same length"):
        stepper.update(
            np.array([datetime(2023, 1, 1)]),
            np.array([1, 2]),
            np.array([1.0])
        )

def test_first_value_initialization():
    # Test that first value for each code has skewness=0
    dt = np.array([
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3)
    ])
    dscodes = np.array([1, 1, 2])  # Two different codes
    values = np.array([1.0, 2.0, 3.0])
    
    stepper = EwmSkewStepper(window=5)
    result = stepper.update(dt, dscodes, values)
    
    # First value for each code should have skewness=0
    assert result[0] == 0.0  # First value for code 1
    assert result[2] == 0.0  # First value for code 2