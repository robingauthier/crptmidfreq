import pytest
import pandas as pd
import numpy as np

from crptmidfreq.season.hols import generate_event_calendar_with_reverse

# pytest ./crptmidfreq/season/tests/test_hols_rev.py --pdb --maxfail=1


def test_hols():
    """

    Returns a table with 
    date        date_p1y    evtname  
    2025-04-20  2024-03-31  Easter   
    2025-03-30  2024-04-21  Easter-R     ### 2025-03-30 = 2024-03-31 + 52 weeks  and 2024-04-21 = 2025-04-20 - 52 weeks
    """
    # Create sample data for two stocks A and B, same weekday one year apart
    df = generate_event_calendar_with_reverse(2010, 2025)
    dfloc = df.loc[df['evtname'].str.contains('Easter')]
    assert dfloc.loc[lambda x:(x['date_p1y'] == '2025-04-20') & (x['date'] == '2024-03-31')].shape[0] == 1
    assert dfloc.loc[lambda x:(x['date_p1y'] == '2025-03-30') & (x['date'] == '2024-04-21')].shape[0] == 1

    dfloc = df.loc[df['evtname'].str.contains('Valentine')]
    assert dfloc.loc[lambda x:(x['date_p1y'] == '2025-02-14') & (x['date'] == '2024-02-14')].shape[0] == 1
    assert dfloc.loc[lambda x:(x['date_p1y'] == '2025-02-12') & (x['date'] == '2024-02-16')].shape[0] == 1
