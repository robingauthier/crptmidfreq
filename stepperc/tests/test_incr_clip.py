import numpy as np

from ..incr_clip import ClipStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_clip.py --pdb --maxfail=1


def test_ffill_stepper_update():
    stepper = ClipStepper(high_clip=3)

    dt = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
    dscode = np.array([1, 1, 2])
    serie = np.array([10, np.nan, -20], dtype=np.float64)

    result = stepper.update(dt, dscode, serie)
    expected = np.array([3, np.nan, -20], dtype=np.float64)

    np.testing.assert_array_equal(result, expected)
