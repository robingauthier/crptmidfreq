import numpy as np

from crptmidfreq.utils.common import clean_folder

from ..incr_pfp import PfPStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_pfp.py --pdb --maxfail=1


def test_pfp_stepper_update():
    clean_folder('test_pfp')
    stepper = PfPStepper(folder='test_pfp', tick=2.0, nbrev=3)

    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

    res_price, res_dir, res_el, res_perf, res_perf2, res_dur = stepper.update(dt, dscode, serie)
    expected_dir = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)
    np.testing.assert_array_equal(res_dir, expected_dir)


def test_pfp_stepper_save_and_load(tmp_path):
    clean_folder('test_pfp')
    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)
    c = 5
    stepper1 = PfPStepper.load(folder='test_pfp', name='', tick=2.0, nbrev=3)
    res_price1, res_dir1, res_el1, res_perf1, res_perf2, res_dur1 = stepper1.update(dt[:c], dscode[:c], serie[:c])
    stepper1.save()
    stepper2 = PfPStepper.load(folder='test_pfp', name='')
    res_price2, res_dir2, res_el2, res_perf3, res_perf4, res_dur2 = stepper2.update(dt[c:], dscode[c:], serie[c:])
    res_dir = np.concatenate([res_dir1, res_dir2])
    expected_dir = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)
    np.testing.assert_array_equal(res_dir, expected_dir)
