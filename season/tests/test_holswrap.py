import pandas as pd

from crptmidfreq.season.holswrap import event_distances

# pytest ./crptmidfreq/season/tests/test_holswrap.py --pdb --maxfail=1


def test_holswrap():
    # Create sample data for two stocks A and B, same weekday one year apart
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2025-01-01')
    df = event_distances(start_date, end_date)

    # after Easter 2024
    assert df.loc[lambda x:x['date'] == '2024-04-06', 'dist_since_last_High'].iloc[0] == 6
    assert df.loc[lambda x:x['date'] == '2024-03-25', 'dist_to_next_High'].iloc[0] == -6
