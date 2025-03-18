import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper


@njit
def groupby_last_values(codes, values, timestamps, last_values, last_timestamps,position):
    """
    equivalent to groupby(['dscode','dt']).last() usuful for resampling the data
    """
    # counter is the actual i in the new non duplicated result
    counter = np.int64(-1)
    counter_loc = np.int64(-1)

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
            raise ValueError(f"DateTime must be strictly increasing per code {ts} last_ts={last_ts} i={i}")
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


class GroupbyLastStepper(BaseStepper):
    """Last value // removes duplicates """

    def __init__(self, folder='', name=''):
        """
        Initialize FfillStepper for forward filling missing values

        Args:
            folder: folder for saving/loading state
            name: name for saving/loading state
        """
        super().__init__(folder,name)

        # Initialize empty state
        self.last_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        # stores the index in the non duplicated table
        self.last_position = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return GroupbyLastStepper.load_utility(cls,folder=folder,name=name)

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
        self.validate_input(dt,dscode,serie)
        
        # Update values and timestamps using numba function
        result_ts, result_code, result = groupby_last_values(
            dscode, serie, dt.view(np.int64),
            self.last_values, self.last_timestamps,self.last_position
        )
        return result_ts, result_code, result
