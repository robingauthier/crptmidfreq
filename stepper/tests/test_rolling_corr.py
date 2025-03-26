import numpy as np

from crptmidfreq.utils.common import clean_folder
from ..rolling_corr import RollingCorrStepper

# pytest ./crptmidfreq/stepper/tests/test_rolling_corr.py --pdb --maxfail=1


def test_rcor_stepper_update():
    clean_folder('test_corr')
    stepper = RollingCorrStepper(folder='test_corr', window=10)

    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie1 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)
    serie2 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, -13.0, -15.0, -20.0, -110.0], dtype=np.float64)

    res_corr = stepper.update(dt, dscode, serie1, serie2)
    expected = np.array([1., 1., 1., 1., 1.,
                         1., 1., 1., 0.01705065, -0.41257552,
                         -0.74521172, -0.09560928], dtype=np.float64)
    np.testing.assert_array_almost_equal(res_corr, expected)


def test_rcor_stepper_save_and_load(tmp_path):
    clean_folder('test_corr')
    c = 5
    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie1 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)
    serie2 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, -13.0, -15.0, -20.0, -110.0], dtype=np.float64)

    stepper1 = RollingCorrStepper.load(folder='test_corr', window=10)
    res_corr1 = stepper1.update(dt[:c], dscode[:c], serie1[:c], serie2[:c])
    stepper1.save()
    stepper2 = RollingCorrStepper.load(folder='test_corr')
    res_corr2 = stepper2.update(dt[c:], dscode[c:], serie1[c:], serie2[c:])
    res_corr = np.concatenate([res_corr1, res_corr2])
    expected = np.array([1., 1., 1., 1., 1.,
                         1., 1., 1., 0.01705065, -0.41257552,
                         -0.74521172, -0.09560928], dtype=np.float64)
    np.testing.assert_array_almost_equal(res_corr, expected)
