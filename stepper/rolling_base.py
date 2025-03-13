import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from .base_stepper import BaseStepper


@njit
def update_rolling_values(timestamps,
                          dscode,
                          values,
                          position,
                          rolling_dict,
                          last_timestamps,
                          window):
    """
    Update rolling values for each code.

    Args:
        dscode: array of categorical codes
        values: array of values to process
        rolling_dict: Dict of deques for each code
        window: Rolling window size

    Returns:
        Updated rolling_dict
    """

    for i in range(len(dscode)):
        code = dscode[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        if code not in position:
            position[code] = 0

        if code not in rolling_dict:
            rolling_dict[code] = np.empty(window, dtype=np.float64)
            for i in range(window):
                rolling_dict[code][i] = np.nan

        position_loc = position[code]
        rolling_dict[code][position_loc] = value
        last_timestamps[code] = ts
        position[code] = (position_loc + 1) % window


class RollingStepper(BaseStepper):
    def __init__(self, folder='', name='', window=1):
        super().__init__(folder,name)
        self.window = window
        self.position = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.values = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_ts = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, window=1):
        """Load instance from saved state or create new if not exists"""
        return RollingStepper.load_utility(cls,folder=folder,name=name,window=window)


    def update_memory(self, dt, dscode, values):
        """
        """
        self.validate_input(dt,dscode,values)
        update_rolling_values(dt.view(np.int64), dscode, values,
                              self.position, self.values, self.last_ts, self.window)
