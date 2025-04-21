import numpy as np
import pandas as pd

from crptmidfreq.utils.common import clean_folder

from ..incr_groupby_last import GroupbyLastStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_groupby_last.py --pdb --maxfail=1


def test_groupby_last_stepper_update():
    folder = "test_groupbylast"
    clean_folder(folder=folder)
    dt = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 7, 8], dtype='int64')
    dscode = np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1])
    serie = np.float64(np.arange(0, 21))
    
    
    df = pd.DataFrame({'dt': dt, 'dscode': dscode, 'serie': serie})
    ldf = df \
        .drop_duplicates(subset=['dt', 'dscode'], keep='last') \
        .sort_values(['dt', 'dscode'])
    stepper = GroupbyLastStepper.load(folder=folder, name='')
    result_ts, result_code, result = stepper.update(dt, dscode, serie)
    rdf = pd.DataFrame({'dt': result_ts, 'dscode': result_code, 'serie': result}) \
        .sort_values(['dt', 'dscode'])
    np.testing.assert_array_equal(ldf['dt'].values, rdf['dt'].values)
    np.testing.assert_array_equal(ldf['dscode'].values, rdf['dscode'].values)
    np.testing.assert_array_equal(ldf['serie'].values, rdf['serie'].values)


def test_groupby_last_stepper_save_and_load(tmp_path):
    folder = "test_groupbylast"
    clean_folder(folder=folder)
    dt = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 7, 8], dtype='int64')
    dscode = np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1])
    serie = np.float64(np.arange(0, 21))
    df = pd.DataFrame({'dt': dt, 'dscode': dscode, 'serie': serie})
    ldf = df \
        .drop_duplicates(subset=['dt', 'dscode'], keep='last') \
        .sort_values(['dt', 'dscode'])
    c = 11
    stepper1 = GroupbyLastStepper.load(folder=folder, name='')
    result_ts1, result_code1, result1 = stepper1.update(dt[:c], dscode[:c], serie[:c])
    stepper1.save()
    stepper2 = GroupbyLastStepper.load(folder=folder, name='')
    result_ts2, result_code2, result2 = stepper2.update(dt[c:], dscode[c:], serie[c:])
    rdf = pd.DataFrame({'dt': np.concatenate([result_ts1, result_ts2]),
                        'dscode': np.concatenate([result_code1, result_code2]),
                        'serie': np.concatenate([result1, result2])}) \
        .sort_values(['dt', 'dscode'])
    np.testing.assert_array_equal(ldf['dt'].values, rdf['dt'].values)
    np.testing.assert_array_equal(ldf['dscode'].values, rdf['dscode'].values)
    np.testing.assert_array_equal(ldf['serie'].values, rdf['serie'].values)
