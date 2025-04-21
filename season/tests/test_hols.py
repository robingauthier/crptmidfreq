import pandas as pd

from crptmidfreq.season.hols import generate_event_calendar

# pytest ./crptmidfreq/season/tests/test_hols.py --pdb --maxfail=1


def test_hols():
    # Create sample data for two stocks A and B, same weekday one year apart
    df = generate_event_calendar(2010, 2025)

    # https://fr.wikipedia.org/wiki/Super_Bowl
    dfloc = df[df['evtname'] == 'SuperBowl']
    assert pd.to_datetime('2024-02-11') in dfloc['date'].tolist()
    assert pd.to_datetime('2025-02-09') in dfloc['date'].tolist()
    assert pd.to_datetime('2019-02-03') in dfloc['date'].tolist()
    assert pd.to_datetime('2020-02-02') in dfloc['date'].tolist()
    assert pd.to_datetime('2021-02-07') in dfloc['date'].tolist()
    assert pd.to_datetime('2022-02-13') in dfloc['date'].tolist()
    assert pd.to_datetime('2023-02-12') in dfloc['date'].tolist()
    dfloc = df[df['evtname'] == 'Easter']
    assert pd.to_datetime('2025-04-20') in dfloc['date'].tolist()
    assert pd.to_datetime('2024-03-31') in dfloc['date'].tolist()
    assert pd.to_datetime('2023-04-09') in dfloc['date'].tolist()
    assert pd.to_datetime('2022-04-17') in dfloc['date'].tolist()
    assert pd.to_datetime('2021-04-04') in dfloc['date'].tolist()
    dfloc = df[df['evtname'] == 'Thanksgiving']
    assert pd.to_datetime('2021-11-25') in dfloc['date'].tolist()
    assert pd.to_datetime('2023-11-30') in dfloc['date'].tolist()
    assert pd.to_datetime('2024-11-28') in dfloc['date'].tolist()
    assert pd.to_datetime('2025-11-27') in dfloc['date'].tolist()
