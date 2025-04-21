from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..incr_timebar import TimeBarStepper


def test_timebar_basic():
    # Generate test data
    base_dt = datetime(2023, 1, 1, 10, 0, 0)  # 10:00:00
    dt = np.array([
        base_dt,
        base_dt + timedelta(seconds=30),
        base_dt + timedelta(minutes=1),
        base_dt + timedelta(minutes=1, seconds=30),
        base_dt + timedelta(minutes=2),
        base_dt + timedelta(minutes=2,seconds=10),
        base_dt + timedelta(minutes=2,seconds=20),
        base_dt + timedelta(minutes=2,seconds=30),
        base_dt + timedelta(minutes=3, seconds=30),
    ])

    dscodes = np.int64(np.ones(len(dt)))
    values = 10+np.float64(np.arange(len(dt)))
    
    # Create stepper
    stepper = TimeBarStepper(period='1T')
    new_bars = stepper.update(dt, dscodes, values)
    full_bars = stepper.get_all_bars()
    fdf=pd.DataFrame(full_bars)
    fdf['date']=pd.to_datetime(fdf['date'])
    assert fdf.shape[0]==4
    assert fdf['cnt'].sum()==9
    fdf['open_ref']=np.array([10,12,14,18])
    fdf['close_ref'] = np.array([11, 13, 17, 18])
    fdf['high_ref'] = np.array([11, 13, 17, 18])
    assert (fdf['close_ref']-fdf['close']).abs().max()==0
    assert (fdf['close_ref'] - fdf['close']).abs().max() == 0
    assert (fdf['high_ref'] - fdf['high']).abs().max() == 0


def test_timebar_multi_dscode():
    # Generate test data
    base_dt = datetime(2023, 1, 1, 10, 0, 0)  # 10:00:00
    dt1 = np.array([
        base_dt,
        base_dt + timedelta(seconds=30),
        base_dt + timedelta(minutes=1),
        base_dt + timedelta(minutes=1, seconds=30),
        base_dt + timedelta(minutes=2),
        base_dt + timedelta(minutes=2, seconds=10),
        base_dt + timedelta(minutes=2, seconds=20),
        base_dt + timedelta(minutes=2, seconds=30),
        base_dt + timedelta(minutes=3, seconds=30),
    ])
    dt2 = np.array([
        base_dt,
        base_dt + timedelta(minutes=1),
        base_dt + timedelta(minutes=1, seconds=30),
        base_dt + timedelta(minutes=2, seconds=30),
        base_dt + timedelta(minutes=3, seconds=30),
    ])

    dscodes1 = np.int64(np.ones(len(dt1)))
    values1 = 10 + np.float64(np.arange(len(dt1)))
    dscodes2 = np.int64(np.ones(len(dt2))*2)
    values2 = 100 + np.float64(np.arange(len(dt2)))

    df=pd.DataFrame({'dt':np.append(dt1,dt2),
                     'dscode':np.append(dscodes1,dscodes2),
                     'val':np.append(values1,values2),
                     }).sort_values('dt')

    # Create stepper
    stepper = TimeBarStepper(period='1T')
    new_bars = stepper.update(df['dt'].values, df['dscode'].values, df['val'].values)
    full_bars = stepper.get_all_bars()
    fdf = pd.DataFrame(full_bars)
    fdf['date'] = pd.to_datetime(fdf['date'])
    fdf=fdf.loc[lambda x:x['dscode']==1]
    assert fdf.shape[0] == 4
    assert fdf['cnt'].sum() == 9
    fdf['open_ref'] = np.array([10, 12, 14, 18])
    fdf['close_ref'] = np.array([11, 13, 17, 18])
    fdf['high_ref'] = np.array([11, 13, 17, 18])
    assert (fdf['close_ref'] - fdf['close']).abs().max() == 0
    assert (fdf['close_ref'] - fdf['close']).abs().max() == 0
    assert (fdf['high_ref'] - fdf['high']).abs().max() == 0


def test_timebar_update():
    # Generate test data
    base_dt = datetime(2023, 1, 1, 10, 0, 0)  # 10:00:00
    dt = np.array([
        base_dt,
        base_dt + timedelta(seconds=30),
        base_dt + timedelta(minutes=1),
        base_dt + timedelta(minutes=1, seconds=30),
        base_dt + timedelta(minutes=2),
        base_dt + timedelta(minutes=2, seconds=10),
        base_dt + timedelta(minutes=2, seconds=20),
        base_dt + timedelta(minutes=2, seconds=30),
        base_dt + timedelta(minutes=3, seconds=30),
    ])
    dscodes = np.int64(np.ones(len(dt)))
    values = 10 + np.float64(np.arange(len(dt)))

    c=5
    # Create stepper
    stepper = TimeBarStepper(folder='test_data', name='test_timebar',period='1T')
    new_bars = stepper.update(dt[:c], dscodes[:c], values[:c])
    stepper.save()

    load_stepper = TimeBarStepper.load(folder='test_data', name='test_timebar')
    assert load_stepper.period_ns==stepper.period_ns
    new_bars2=load_stepper.update(dt[c:], dscodes[c:], values[c:])

    full_bars = load_stepper.get_all_bars()
    fdf = pd.DataFrame(full_bars)
    fdf['date'] = pd.to_datetime(fdf['date'])
    assert fdf.shape[0] == 4
    assert fdf['cnt'].sum() == 9
    fdf['open_ref'] = np.array([10, 12, 14, 18])
    fdf['close_ref'] = np.array([11, 13, 17, 18])
    fdf['high_ref'] = np.array([11, 13, 17, 18])
    assert (fdf['close_ref'] - fdf['close']).abs().max() == 0
    assert (fdf['close_ref'] - fdf['close']).abs().max() == 0
    assert (fdf['high_ref'] - fdf['high']).abs().max() == 0


    fdf2 = pd.DataFrame(new_bars2)
    fdf2['date'] = pd.to_datetime(fdf2['date'])
    assert fdf2.shape[0]==1