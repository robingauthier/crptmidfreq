import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from config_loc import get_data_folder


@njit
def groupby_last_values(codes, values, timestamps, last_values, last_timestamps):
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
    counter = np.int64(-1)
    counter_loc = np.int64(-1)
    position = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    result = np.empty(len(codes), dtype=np.float64)
    result_ts = np.empty(len(codes), dtype=np.int64)
    result_code = np.empty(len(codes), dtype=np.int64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:
            print(ts)
            print(code)
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing per code")
        if ts != last_ts:
            position[code] = np.int64(counter + 1)
            counter = np.int64(counter + 1)
        if code not in position:
            position[code] = counter
        # do not use Dict.get() it creates many issues in Numba
        counter_loc = position[code]
        if counter_loc < 0:
            counter_loc = 0
        counter_loc = np.int64(np.float64(counter_loc))
        result[counter_loc] = value
        result_code[counter_loc] = code
        result_ts[counter_loc] = ts

        # Store result and update last known value
        last_values[code] = value
        last_timestamps[code] = ts

    result_ts = result_ts[:counter + 1]
    result_code = result_code[:counter + 1]
    result = result[:counter + 1]
    return result_ts, result_code, result


class GroupbyLastStepper:
    """Forward fill stepper that maintains last known values per code"""

    def __init__(self, folder='', name=''):
        """
        Initialize FfillStepper for forward filling missing values

        Args:
            folder: folder for saving/loading state
            name: name for saving/loading state
        """
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name

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
        """Save internal state to file"""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        state = {
            'last_timestamps': dict(self.last_timestamps),
            'last_values': dict(self.last_values)
        }

        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state"""
        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            return cls(folder=folder, name=name)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create new instance
        instance = cls(folder=folder, name=name)

        # Convert regular dicts back to numba Dicts
        for k, v in state['last_values'].items():
            instance.last_values[k] = v
        for k, v in state['last_timestamps'].items():
            instance.last_timestamps[k] = v

        return instance

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
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        if not dt.dtype == 'int64':
            # Convert datetime64 to int64 nanoseconds for Numba
            timestamps = dt.astype('datetime64[ns]').astype('int64')
        else:
            timestamps = dt

        # Update values and timestamps using numba function
        result_ts, result_code, result = groupby_last_values(
            dscode, serie, timestamps,
            self.last_values, self.last_timestamps
        )
        self.save()
        return result_ts, result_code, result
