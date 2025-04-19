import pandas as pd
from datetime import date, timedelta
from dateutil.easter import easter
from pandas.tseries.offsets import WeekOfMonth, DateOffset
import convertdate.hebrew as hebrew
import convertdate.islamic as islamic
import convertdate.holidays as chols
from dateutil.relativedelta import relativedelta, SU


def todate(tupleymd):
    return date(tupleymd[0], tupleymd[1], tupleymd[2])


def get_superbowl(year):
    """
    For a given NFL season (e.g. 2010),
    returns the date of its Super Bowl,
    which is the first Sunday (weekday=6) of Feb in season+1.
    """
    if year <= 2021:
        return date(year, 2, 1)+relativedelta(weekday=SU)
    else:
        return date(year, 2, 1)+relativedelta(weekday=SU, weeks=1)


def generate_event_calendar_loc(y):
    rows = [
        (date(y,  1,  1), 'NewYear',              'High'),
        (todate(chols.christmas(y)), 'Christmas',   'High'),
        (todate(chols.labor_day(y)), 'LaborDay',   'Mid'),
        (todate(chols.juneteenth(y)), 'JuneTeenth',   'Mid'),
        (todate(chols.easter(y)), 'Easter',   'High'),
        (todate(chols.thanksgiving(y)), 'Thanksgiving',   'High'),
        (todate(chols.mothers_day(y)), 'MotherDay',   'Mid'),
        (todate(chols.memorial_day(y)), 'MemorialDay',   'Mid'),
        (todate(chols.eid_alfitr(y)), 'EidAlfitr',   'Mid'),
        (date(y, 10, 31), 'Halloween', 'Mid'),
        (date(y,  2, 14), 'ValentinesDay',       'Mid'),
        (get_superbowl(y), 'SuperBowl',          'Mid'),
        (date(y,  3, 17), 'StPatricksDay',       'Mid'),
        (date(y,  5,  5), 'CincoDeMayo',       'Mid'),


    ]
    return rows


def generate_event_calendar(start_year=2010, end_year=2030):

    rows = []
    for y in range(start_year, end_year + 1):
        rows += generate_event_calendar_loc(y)

    df = pd.DataFrame(rows)
    df.columns = ['date', 'evtname', 'evtimp']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return df
