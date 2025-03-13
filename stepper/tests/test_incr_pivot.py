import numpy as np

from utils.common import clean_folder
from ..incr_pivot import PivotStepper

# pytest ./stepper/tests/test_incr_pfp.py --pdb --maxfail=1

def test_pivot_stepper_update():
    clean_folder('test_pfp')
    stepper = PivotStepper(folder='test_pfp')

    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

    res_price, res_dir, res_el, res_perf, res_perf2, res_dur = stepper.update(dt, dscode, serie)
    expected_dir = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)
    np.testing.assert_array_equal(res_dir, expected_dir)
