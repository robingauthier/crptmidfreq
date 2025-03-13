import numpy as np
import pytest

from stepper.incr_merge_asof import MergeAsofStepper

# pytest ./stepper/tests/test_incr_merge_asof.py --pdb --maxfail=1


def test_merge_asof_stepper_basic():
    # Create stepper
    stepper = MergeAsofStepper(folder='test', name='basic')

    # First batch
    time1 = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    dscode1 = np.array([1, 1, 1, 1, 1], dtype=np.int64)

    time2 = np.array([2, 4, 6], dtype=np.int64)
    dscode2 = np.array([1, 1, 1], dtype=np.int64)
    value2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    merged_value2 = stepper.update(
        time1, dscode1, time2, dscode2, value2
    )
    print(stepper.right_timestamps)
    print(stepper.right_values)
    # Expected results for first batch
    expected_value2 = np.array([np.nan, 1.0, 2.0, 3.0, 3.0], dtype=np.float64)
    np.testing.assert_array_almost_equal(merged_value2, expected_value2)

    # Second batch - should use historical data
    time1_b2 = np.array([10, 11, 14, 16], dtype=np.int64)
    dscode1_b2 = np.array([1, 1, 1, 1], dtype=np.int64)

    time2_b2 = np.array([12, 13, 15, 17], dtype=np.int64)
    dscode2_b2 = np.array([1, 1, 1, 1], dtype=np.int64)
    value2_b2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64)

    merged_value2 = stepper.update(
        time1_b2, dscode1_b2, time2_b2, dscode2_b2, value2_b2
    )

    # Expected results for second batch
    expected_value2 = np.array([3.0, 3.0, 5.0, 6.0], dtype=np.float64)
    np.testing.assert_array_almost_equal(merged_value2, expected_value2)


def test_merge_asof_stepper_timestamp():
    # Create stepper
    stepper = MergeAsofStepper(folder='test', name='basic')

    # First batch
    time1 = np.array([1, 3, 1, 7, 9], dtype=np.int64)
    dscode1 = np.array([1, 1, 1, 1, 1], dtype=np.int64)

    time2 = np.array([2, 4, 6], dtype=np.int64)
    dscode2 = np.array([1, 1, 1], dtype=np.int64)
    value2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(AssertionError):
        merged_value2 = stepper.update(
            time1, dscode1, time2, dscode2, value2
        )
    stepper = MergeAsofStepper(folder='test', name='basic')

    # First batch
    time1 = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    dscode1 = np.array([1, 1, 1, 1, 1], dtype=np.int64)

    time2 = np.array([2, 4, 3], dtype=np.int64)
    dscode2 = np.array([1, 1, 1], dtype=np.int64)
    value2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(AssertionError):
        merged_value2 = stepper.update(
            time1, dscode1, time2, dscode2, value2
        )


def test_merge_asof_stepper_dscode():
    # Create stepper
    stepper = MergeAsofStepper(folder='test', name='basic')

    # First batch
    time1 = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    dscode1 = np.array([1, 2, 1, 2, 1], dtype=np.int64)

    time2 = np.array([2, 4, 6], dtype=np.int64)
    dscode2 = np.array([1, 2, 1], dtype=np.int64)
    value2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    merged_value2 = stepper.update(
        time1, dscode1, time2, dscode2, value2
    )
    expected_value2 = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
    np.testing.assert_array_almost_equal(merged_value2, expected_value2)

    # Second batch - should use historical data
    time1_b2 = np.array([10, 11, 14, 16], dtype=np.int64)
    dscode1_b2 = np.array([1, 1, 2, 1], dtype=np.int64)
    time2_b2 = np.array([12, 13, 15, 17], dtype=np.int64)
    dscode2_b2 = np.array([1, 2, 2, 1], dtype=np.int64)
    value2_b2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    merged_value2 = stepper.update(
        time1_b2, dscode1_b2, time2_b2, dscode2_b2, value2_b2
    )
    # Expected results for second batch
    expected_value2 = np.array([3.0, 3.0, 5.0, 4.0], dtype=np.float64)
    np.testing.assert_array_almost_equal(merged_value2, expected_value2)
