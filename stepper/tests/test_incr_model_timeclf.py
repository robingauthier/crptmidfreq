import numpy as np
import numpy as np
import pandas as pd
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.stepper.incr_model_timeclf import TimeClfStepper
# pytest ./crptmidfreq/stepper/tests/test_incr_model_timeclf.py --pdb --maxfail=1


def test_time_clf_stepper_update():
    # Clean the test folder
    clean_folder('test_time_clf')

    # Instantiate the stepper with small parameters for testing purposes.
    # (Using lookback=10, minlookback=5, fitfreq=3, gap=1)
    stepper = TimeClfStepper(folder='test_time_clf', name='test',
                             lookback=10, minlookback=5, fitfreq=3, gap=1)

    dts = np.int64(np.arange(0, 30, 1)/2)
    #dts = dts+pd.to_datetime('2022-01-01').timestamp()*1e6
    dscode = np.ones_like(dts, dtype=np.int64)
    np.random.seed(42)
    serie = np.random.rand(30, 1)

    # Call update, which returns a result array (of predictions)
    stepper.update(dts, dscode, serie)

    timedf = pd.DataFrame([{k: v for k, v in x.items() if k.endswith('_i')} for x in stepper.ltimes])
    assert timedf['train_stop_i'].is_monotonic_increasing
    assert np.all(timedf['train_stop_i'] <= timedf['test_start_i'])

    time2df = pd.DataFrame([{k: v for k, v in x.items() if k.endswith('_dt')} for x in stepper.ltimes])
    assert time2df['train_stop_dt'].is_monotonic_increasing
    assert np.all(time2df['train_stop_dt'] <= time2df['test_start_dt'])


def test_time_clf_stepper_update_2():
    # Clean the test folder
    clean_folder('test_time_clf')

    # Instantiate the stepper with small parameters for testing purposes.
    # (Using lookback=10, minlookback=5, fitfreq=3, gap=1)
    stepper = TimeClfStepper(folder='test_time_clf', name='test',
                             lookback=10, minlookback=5, fitfreq=3, gap=1)

    dts = np.int64(np.arange(0, 30, 1)/2)
    #dts = dts+pd.to_datetime('2022-01-01').timestamp()*1e6
    dscode = np.ones_like(dts, dtype=np.int64)
    np.random.seed(42)
    serie = np.random.rand(30, 1)

    # Call update, which returns a result array (of predictions)
    stepper.update(dts, dscode, serie)
    timedf = pd.DataFrame([{k: v for k, v in x.items() if k.endswith('_dt')} for x in stepper.ltimes])

    dts = np.int64(np.arange(31, 32, 1)/2)
    #dts = dts+pd.to_datetime('2022-01-01').timestamp()*1e6
    dscode = np.ones_like(dts, dtype=np.int64)
    np.random.seed(42)
    serie = np.random.rand(30, 1)
    stepper.update(dts, dscode, serie)

    timedf2 = pd.DataFrame([{k: v for k, v in x.items() if k.endswith('_dt')} for x in stepper.ltimes])
    assert timedf2.shape[0] == timedf.shape[0]+1
