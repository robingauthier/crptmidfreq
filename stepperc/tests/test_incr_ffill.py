import numpy as np
import pytest
from numba import types
from numba.typed import Dict

from ..incr_ffill import FfillStepper, ffill_values

# pytest ./crptmidfreq/stepper/tests/test_incr_ffill.py --pdb --maxfail=1


def test_ffill_values_basic():
    codes = np.array([1, 1, 2, 1, 2, 3])
    values = np.array([10, np.nan, 20, 15, np.nan, 30], dtype=np.float64)
    timestamps = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

    # Use numba types for the Dict
    last_values = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_timestamps = Dict.empty(key_type=types.int64, value_type=types.int64)

    result = ffill_values(codes, values, timestamps, last_values, last_timestamps)

    expected = np.array([10, 10, 20, 15, 20, 30], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_ffill_values_with_empty_memory():
    codes = np.array([1, 2])
    values = np.array([np.nan, 20], dtype=np.float64)
    timestamps = np.array([1, 2], dtype=np.int64)

    last_values = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_timestamps = Dict.empty(key_type=types.int64, value_type=types.int64)

    result = ffill_values(codes, values, timestamps, last_values, last_timestamps)

    expected = np.array([np.nan, 20], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_ffill_values_invalid_timestamp_order():
    codes = np.array([1, 1])
    values = np.array([10, 15], dtype=np.float64)
    timestamps = np.array([2, 1], dtype=np.int64)

    last_values = Dict.empty(key_type=types.int64, value_type=types.float64)
    last_timestamps = Dict.empty(key_type=types.int64, value_type=types.int64)

    with pytest.raises(ValueError, match="DateTime must be strictly increasing per code"):
        ffill_values(codes, values, timestamps, last_values, last_timestamps)


# Test for FfillStepper class
def test_ffill_stepper_update():
    stepper = FfillStepper()

    dt = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
    dscode = np.array([1, 1, 2])
    serie = np.array([10, np.nan, 20], dtype=np.float64)

    result = stepper.update(dt, dscode, serie)
    expected = np.array([10, 10, 20], dtype=np.float64)

    np.testing.assert_array_equal(result, expected)


def test_ffill_stepper_with_empty_inputs():
    stepper = FfillStepper()

    dt = np.array([], dtype='datetime64[D]')
    dscode = np.array([], dtype=np.int64)
    serie = np.array([], dtype=np.float64)

    result = stepper.update(dt, dscode, serie)
    expected = np.array([], dtype=np.float64)

    np.testing.assert_array_equal(result, expected)
