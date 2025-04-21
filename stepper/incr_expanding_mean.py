import numpy as np
from numba import njit, types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger

logger = get_logger()


@njit(cache=True)
def mean_values(codes, values, timestamps, last_sum, last_cnt, last_timestamps):
    """
    Expanding Mean
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
        if code in last_sum:
            last_sum[code] = last_sum[code]+value
            last_cnt[code] = last_cnt[code]+1
        else:
            last_sum[code] = value
            last_cnt[code] = 1
        result[i] = last_sum[code]/last_cnt[code]
        last_timestamps[code] = ts
    return result


class ExpandingMeanStepper(BaseStepper):
    """
    Performs an expanding maximum
    """

    def __init__(self, folder='', name=''):
        """

        """
        super().__init__(folder, name)
        # Initialize empty state
        self.last_sum = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_cnt = Dict.empty(
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
        return ExpandingMeanStepper.load_utility(cls, folder=folder, name=name)

    def update(self, dt, dscode, serie):
        """

        """
        self.validate_input(dt, dscode, serie)
        res = mean_values(dscode, serie, dt.view(np.int64),
                          self.last_sum, self.last_cnt, self.last_timestamps)
        return res
