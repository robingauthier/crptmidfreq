import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger

logger = get_logger()


@njit(cache=True)
def min_values(codes, values, timestamps, last_values, last_timestamps):
    """
    Expanding Max
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

        # Store result and update last known value
        if code in last_values:
            current_max = last_values[code]
            new_max = min(current_max, value)
            result[i] = new_max
        else:
            new_max = value
            result[i] = value
        last_values[code] = new_max
        last_timestamps[code] = ts
    return result


class MinStepper(BaseStepper):
    """
    Performs an expanding maximum
    """

    def __init__(self, folder='', name=''):
        """

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
    def load(cls, folder, name, n_buckets=8, freq=int(60*24*5)):
        """Load instance from saved state or create new if not exists"""
        return MinStepper.load_utility(cls, folder=folder, name=name)

    def update(self, dt, dscode, serie):
        """

        """
        self.validate_input(dt, dscode, serie)
        res = min_values(dscode, serie, dt.view(np.int64),
                         self.last_values, self.last_timestamps)
        return res
