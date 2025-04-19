import pytest
import pandas as pd
import numpy as np

from crptmidfreq.season.yoy import deseasonalize_yoy  # replace with actual import path

# pytest ./crptmidfreq/season/tests/test_yoy.py --pdb --maxfail=1


def test_deseasonalize_yoy_basic():
    # Create sample data for two stocks A and B, same weekday one year apart
    df = pd.DataFrame({
        'date': pd.to_datetime([
            '2020-01-03', '2021-01-01',  # Stock A: both Mondays
            '2020-01-06', '2021-01-04',  # Stock A: both Mondays
            '2020-01-07', '2021-01-05',   # Stock B: both Tuesdays
            '2021-01-08', '2021-01-09', '2021-01-10',
        ], format='%Y-%m-%d'),
        'dscode': ['A', 'A',
                   'A', 'A',
                   'B', 'B',
                   'B', 'B', 'B'],
        'serie': [70.0, 80.0,
                  100.0, 110.0,
                  200.0, 180.0,
                  100.0, 110.0, 120.0]
    }).sort_values('date').reset_index(drop=True)

    # Apply deseasonalization
    rdf, _ = deseasonalize_yoy(df, date_col='date', stock_col='dscode', serie_col='serie')
    print(rdf)

    # Expected:
    expected_A = 0.095238
    expected_B = -0.105263

    actual_A = rdf.loc[(rdf['dscode'] == 'A') & (
        rdf['date'] == pd.Timestamp('2021-01-04')), 'serie_yoy_pct'].iloc[0]
    actual_B = rdf.loc[(rdf['dscode'] == 'B') & (
        rdf['date'] == pd.Timestamp('2021-01-05')), 'serie_yoy_pct'].iloc[0]

    assert np.isclose(actual_A, expected_A), f"Expected {expected_A}, got {actual_A}"
    assert np.isclose(actual_B, expected_B), f"Expected {expected_B}, got {actual_B}"


def test_deseasonalize_yoy_basic2():
    # If we call twice it should still work
    df = pd.DataFrame({
        'date': pd.to_datetime([
            '2020-01-03', '2021-01-01',  # Stock A: both Mondays
            '2020-01-06', '2021-01-04',  # Stock A: both Mondays
            '2020-01-07', '2021-01-05',   # Stock B: both Tuesdays
            '2021-01-08', '2021-01-09', '2021-01-10',
        ], format='%Y-%m-%d'),
        'dscode': ['A', 'A',
                   'A', 'A',
                   'B', 'B',
                   'B', 'B', 'B'],
        'serie': [70.0, 80.0,
                  100.0, 110.0,
                  200.0, 180.0,
                  100.0, 110.0, 120.0],
        'serie2': [70.0, 80.0,
                   100.0, 110.0,
                   200.0, 180.0,
                   100.0, 110.0, 120.0]
    }).sort_values('date').reset_index(drop=True)

    # Apply deseasonalization
    rdf, _ = deseasonalize_yoy(df, date_col='date', stock_col='dscode', serie_col='serie')
    rdf, _ = deseasonalize_yoy(df, date_col='date', stock_col='dscode', serie_col='serie2')
