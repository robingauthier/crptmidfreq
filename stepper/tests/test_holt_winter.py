import pandas as pd

from crptmidfreq.stepper.incr_holt_winter import *

# pytest ./crptmidfreq/stepper/tests/test_holt_winter.py --pdb --maxfail=1

def generate_seasonal_data(n_samples, n_codes, season_period=10):
    """
    Generate synthetic data with strong seasonality of a given period.
    Args:
        n_samples (int): Number of time steps in the time series.
        n_codes (int): Number of different codes (e.g., for multiple time series).
        season_period (int): The period of the seasonal cycle (default 10).
    Returns:
        dt (numpy array): Dates of the time series.
        dscode (numpy array): Categorical codes for each time series.
        serie (numpy array): The generated seasonal data with noise.
    """
    np.random.seed(42)
    
    # Generate dates
    dt = pd.date_range('2020-01-01', periods=n_samples, freq='D').to_numpy()
    
    # Generate codes for different time series
    dscode = np.repeat(np.arange(n_codes), n_samples // n_codes)

    # Create strong seasonality with noise
    time = np.arange(n_samples)
    seasonality = np.cos(2 * np.pi * time / season_period)  # Seasonal pattern
    trend = 0.1 * time  # Gradual upward trend
    noise = np.random.normal(scale=2.0, size=n_samples)  # Random noise
    
    # Combine the components: seasonal effect + trend + noise
    serie = 100 + 5 * seasonality + trend + noise  # Base level 10 and amplitude 5 for seasonality

    return dt, dscode, serie,noise

def test_holt_winter_with_seasonality():
    # Generate test data with strong seasonality (period of 10)
    n_samples = 1000
    n_codes = 1  # Single series for simplicity, can be expanded
    season_period = 10  # Strong seasonality with period 10
    dt, dscode, serie,noise = generate_seasonal_data(n_samples, n_codes, season_period)

    # Create HoltWinterStepper instance
    sss = HoltWinterStepper(folder='test_data', name='test_holt_winter', 
                            alpha=100, beta=100, gamma=20, 
                            seasonality=season_period, 
                            time_unit=1e9*3600*24, 
                            use_seasonality=True)
    
    # Run the update method on the generated data
    serier= sss.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'noisec':noise,
        'serie': serie,
        'serier': serier,
        'noiser':serie-serier,
    }).iloc[int(n_samples*0.2):]  # Skip the first 10% of data for better correlation

    
    # For now, just use raw values for the correlation test
    # Compare results using correlation
    correlation = df['noiser'].corr(df['noisec'])

    #to_csv(df,'test_holt_winter.csv')
    
    # Assert a good correlation (you can adjust the threshold as needed)
    assert correlation > 0.6, f"Expected correlation >0.98, got {correlation}"
    
    