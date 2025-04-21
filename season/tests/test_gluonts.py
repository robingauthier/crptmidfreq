import numpy as np
import pandas as pd
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

from crptmidfreq.season.gluonts_v1 import fit_gluonts_model

# pytest ./crptmidfreq/season/tests/test_gluonts.py --pdb --maxfail=1


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
    trend = np.cos(2 * np.pi * time / 100/season_period)  # Gradual upward trend
    noise = np.random.normal(scale=1.0, size=n_samples)  # Random noise

    # Combine the components: seasonal effect + trend + noise
    serie = seasonality + trend + noise  # Base level 10 and amplitude 5 for seasonality

    df = pd.DataFrame({
        'date': dt,
        'dscode': dscode,
        'seasonality': seasonality,
        'trend': trend,
        'serie': serie,
        'ref': serie-noise,
        'noise': noise
    })
    df['dtsi'] = pd.to_datetime(df['date']).astype(np.int64) / 10**9/3600/24
    df = df.sort_values('dtsi')
    return df


def test_fit_gluonts_model():
    # context_length – Number of time steps prior to prediction time that the model takes as inputs (default: 10 * prediction_length)

    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=5,  # forecast_its[ii].samples.shape[1]
        context_length=30,  # forecast_its[ii].samples.shape[0]
        trainer=Trainer(ctx="cpu", epochs=10, learning_rate=1e-1, num_batches_per_epoch=100),
    )
    df = generate_seasonal_data(n_samples=10_000, n_codes=1, season_period=30)

    df['tref'] = df.groupby('dscode')['ref'].transform(lambda x: x.shift(-1))
    fd = {}
    ndf = fit_gluonts_model(df, fd, estimator, target_col='serie')
    correlation = ndf[['gluonts_predh0_mean', 'tref']].corr().iloc[0, 1]
    assert correlation > 0.5, f"Correlation is too low: {correlation}"

    res = ndf['gluonts_predh0_mean'].replace(0.0, np.nan).dropna()
    assert abs(res.quantile(0.8)-df['tref'].quantile(0.8)) < 0.5

    assert abs(res.quantile(0.2)-df['tref'].quantile(0.2)) < 0.5
