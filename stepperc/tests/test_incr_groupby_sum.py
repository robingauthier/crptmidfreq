import numpy as np
import pandas as pd

from crptmidfreq.utils.common import clean_folder
from crptmidfreq.stepper.incr_groupby_sum import GroupbySumStepper


# pytest ./crptmidfreq/stepperc/tests/test_incr_groupby_sum.py --pdb --maxfail=1


def test_groupby_last_stepper_update():
    folder = "test_groupbylast"
    clean_folder(folder=folder)

    n_samples = 1000
    dt = np.int64(np.arange(n_samples)//10)
    dscode = np.repeat([1, 2, 3], n_samples/3)

    dt = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 7, 8], dtype='int64')
    dscode = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1, 1, 1, 2, 3, 4, 1])
    by = np.ones_like(dscode).astype(np.int64)
    serie = np.float64(np.arange(0, 21))
    df = pd.DataFrame({'dt': dt, 'dscode': dscode, 'serie': serie})
    ldf = df \
        .groupby('dt').sum()\
        .sort_index()\
        .reset_index()
    stepper = GroupbySumStepper.load(folder=folder, name='')
    result_ts, result_by, result_val = stepper.update(dt, dscode, by, serie)

    rdf = pd.DataFrame({'dt': result_ts, 'by': result_by, 'serie': result_val}) \
        .sort_values(['dt'])
    np.testing.assert_array_equal(ldf['dt'].values, rdf['dt'].values)
    np.testing.assert_array_equal(ldf['serie'].values, rdf['serie'].values)
