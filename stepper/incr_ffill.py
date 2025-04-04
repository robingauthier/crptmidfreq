import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper


@njit(cache=True)
def ffill_values(codes, values, timestamps, last_values, last_timestamps):
    """
    Forward fill values for each code, using last known values from memory.

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        last_values: Dict mapping codes to last known values
        last_timestamps: Dict mapping codes to last timestamp for each code

    Returns:
        Array of forward-filled values corresponding to each input row
    """
    result = np.empty(len(codes), dtype=np.float64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:
            raise ValueError("DateTime must be strictly increasing per code")

        # If value is NaN, use last known value for this code
        if np.isnan(value):
            value = last_values.get(code, np.nan)

        # Store result and update last known value
        result[i] = value
        if not np.isnan(value):
            last_values[code] = value
            last_timestamps[code] = ts

    return result


class FfillStepper(BaseStepper):
    """Forward fill stepper that maintains last known values per code"""

    def __init__(self, folder='', name=''):
        """
        Initialize FfillStepper for forward filling missing values

        Args:
            folder: folder for saving/loading state
            name: name for saving/loading state
        """
        super().__init__(folder, name)

        # Initialize empty state
        self.last_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, window=1):
        """Load instance from saved state or create new if not exists"""
        return FfillStepper.load_utility(cls, folder=folder, name=name)

    def update(self, dt, dscode, serie):
        """
        Update forward fill state and return filled values

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing forward-filled values
        """
        # Input validation
        self.validate_input(dt, dscode, serie)

        # Update values and timestamps using numba function
        return ffill_values(
            dscode, serie, dt.view(np.int64),
            self.last_values, self.last_timestamps
        )
