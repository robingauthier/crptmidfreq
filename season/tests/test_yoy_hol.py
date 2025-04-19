import pytest
import pandas as pd
import numpy as np

from crptmidfreq.season.yoy_hol import deseasonalize_yoy_hol  # replace with actual import path

# pytest ./crptmidfreq/season/tests/test_yoy_hol.py --pdb --maxfail=1


def test_deseasonalize_yoy_basic():
    # Create sample data for two stocks A and B, same weekday one year apart
    df = pd.DataFrame({
        'date': pd.to_datetime([
            '2020-01-03', '2021-01-01',  # Stock A: both Mondays
            '2020-01-06', '2021-01-04',  # Stock A: both Mondays
            '2020-01-07', '2021-01-05',   # Stock B: both Tuesdays
            '2024-03-31', '2024-04-21',
            '2025-03-30', '2025-04-20',
        ], format='%Y-%m-%d'),
        'dscode': ['A', 'A',
                   'A', 'A',
                   'B', 'B',
                   'B', 'B',
                   'B', 'B'],
        'serie': [70.0, 80.0,
                  100.0, 110.0,
                  200.0, 180.0,
                  100.0, 110.0,
                  120.0, 90.0]
    }).sort_values('date').reset_index(drop=True)

    #assert dfloc.loc[lambda x:(x['date'] == '2025-04-20') & (x['date_p1y'] == '2024-03-31')].shape[0] == 1
    #assert dfloc.loc[lambda x:(x['date'] == '2025-03-30') & (x['date_p1y'] == '2024-04-21')].shape[0] == 1

    # Apply deseasonalization
    rdf, _ = deseasonalize_yoy_hol(df, date_col='date', stock_col='dscode', serie_col='serie')
    print(rdf)

    actual_B = rdf.loc[(rdf['dscode'] == 'B') & (
        rdf['date'] == pd.Timestamp('2025-04-20')), 'serie_yoy'].iloc[0]
    expected_B = 90-100  # 2025-04-20 vs 2024-03-31

    assert np.isclose(actual_B, expected_B), f"Expected {expected_B}, got {actual_B}"

    actual_B = rdf.loc[(rdf['dscode'] == 'B') & (
        rdf['date'] == pd.Timestamp('2025-03-30')), 'serie_yoy'].iloc[0]
    expected_B = 120-110  # 2025-03-30 vs 2024-04-21

    assert np.isclose(actual_B, expected_B), f"Expected {expected_B}, got {actual_B}"
