import numpy as np

from utils.common import clean_folder
from ..incr_cumsum import CumSumStepper


# Test for FfillStepper class
def test_ffill_stepper_update():
    stepper = CumSumStepper()

    dt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    dscode = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    serie = np.array([1, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1], dtype=np.float64)

    result = stepper.update(dt, dscode, serie)
    expected = np.array([1, 1, 2, 3, 3, 1, 2, 2, 3, 4], dtype=np.float64)

    np.testing.assert_array_equal(result, expected)


def test_ffill_stepper_save_and_load(tmp_path):
    folder = "test_cumsum_stepper"
    clean_folder(folder)
    dt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    dscode = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    serie = np.array([1, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1], dtype=np.float64)
    c = 3
    stepper = CumSumStepper(folder=str(folder), name="")
    r1 = stepper.update(dt[:c], dscode[:c], serie[:c])
    loaded_stepper = CumSumStepper.load(folder=str(folder), name="")
    r2 = loaded_stepper.update(dt[c:], dscode[c:], serie[c:])
    expected = np.array([1, 1, 2, 3, 3, 1, 2, 2, 3, 4], dtype=np.float64)
    result = np.concatenate([r1, r2])
    np.testing.assert_array_equal(result, expected)
