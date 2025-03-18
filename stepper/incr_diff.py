import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
from crptmidfreq.stepper.base_stepper import BaseStepper

@njit
def update_diff_values(codes, values, timestamps, diff_values, last_timestamps):
    """
    Vectorized update of difference values and timestamps

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        window: window size for difference calculation
        diff_values: Dict mapping codes to current difference values
        last_timestamps: Dict mapping codes to last timestamp for each code

    Returns:
        Array of difference values corresponding to each input row
    """
    result = np.empty(len(codes), dtype=np.float64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing per code")

        # Update difference value
        old_value = diff_values.get(code, np.nan)
        if np.isnan(old_value):
            new_value = np.nan
        else:
            new_value = value - old_value

        # Store updates
        diff_values[code] = value
        last_timestamps[code] = ts

        # Store result for this row
        result[i] = new_value

    return result


class DiffStepper(BaseStepper):
    
    def __init__(self, folder='', name='', window=1):
        assert window==1
        super().__init__(folder,name)
        self.window = window

        # Initialize empty state
        self.diff_values = Dict.empty(
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
        return DiffStepper.load_utility(cls,folder=folder,name=name,window=window)


    def update(self, dt, dscode, serie):
        """
        Update difference values for each code and return the difference values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing difference values
        """
        # Input validation
        self.validate_input(dt,dscode,serie)
        # Update values and timestamps using numba function
        return update_diff_values(
            dscode, serie, dt.view(np.int64),
            self.diff_values, self.last_timestamps
        )
