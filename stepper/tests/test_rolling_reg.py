import numpy as np

from crptmidfreq.utils.common import clean_folder
from ..rolling_reg import RollingRidgeStepper


# pytest ./stepper/tests/test_rolling_reg.py --pdb --maxfail=1

def test_reg_stepper_update():
    clean_folder('test_corr')
    stepper = RollingRidgeStepper(folder='test_corr', window=10)

    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie1 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)
    serie2 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, -13.0, -15.0, -20.0, -110.0], dtype=np.float64)

    alpha, beta, resid = stepper.update(dt, dscode, serie1, serie2)
    np.testing.assert_almost_equal(beta[3], 1.0)


def test_reg_stepper_save_and_load(tmp_path):
    clean_folder('test_corr')
    c = 5
    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie1 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)
    serie2 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, -13.0, -15.0, -20.0, -110.0], dtype=np.float64)

    stepper1 = RollingRidgeStepper.load(folder='test_corr', window=10)
    a1, b1, r1 = stepper1.update(dt[:c], dscode[:c], serie1[:c], serie2[:c])
    stepper1.save()
    stepper2 = RollingRidgeStepper.load(folder='test_corr')
    a2, b2, r2 = stepper2.update(dt[c:], dscode[c:], serie1[c:], serie2[c:])

    stepper3 = RollingRidgeStepper.load(folder='test_corr3', window=10)
    a3, b3, r3 = stepper3.update(dt, dscode, serie1, serie2)

    np.testing.assert_array_almost_equal(a3, np.concatenate([a1, a2]))
    np.testing.assert_array_almost_equal(r3, np.concatenate([r1, r2]))
