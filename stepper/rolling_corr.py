
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


@njit
def update_rolling_corr(timestamps,
                        dscode,
                        values1,
                        values2,
                        position,
                        rolling_dict1,
                        rolling_dict2,
                        last_timestamps,
                        window):
    """

    """
    result = np.zeros(len(dscode), dtype=np.float64)
    for i in range(len(dscode)):
        code = dscode[i]
        value1 = values1[i]
        value2 = values2[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        if code not in position:
            position[code] = 0

        if code not in rolling_dict1:
            rolling_dict1[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_dict1[code][j] = np.nan
        if code not in rolling_dict2:
            rolling_dict2[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_dict2[code][j] = np.nan

        position_loc = position[code]
        rolling_dict1[code][position_loc] = value1
        rolling_dict2[code][position_loc] = value2
        last_timestamps[code] = ts
        position[code] = (position_loc + 1) % window
        result[i] = np.corrcoef(np.nan_to_num(rolling_dict1[code]),
                                np.nan_to_num(rolling_dict2[code]))[0, 1]
    return result


class RollingCorrStepper(BaseStepper):
    def __init__(self, folder='', name='', window=1):
        super().__init__(folder,name)
        self.window = window
        
        self.position = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.values1 = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.values2 = Dict.empty(
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
    def load(cls, folder, name='', window=1):
        """Load instance from saved state or create new if not exists"""
        return RollingCorrStepper.load_utility(cls,folder=folder,name=name,window=window)

    def update(self, dt, dscode, values1, values2):
        self.validate_input(dt,dscode,values1,serie2=values2)
        res = update_rolling_corr(dt.view(np.int64), dscode, values1, values2,
                                  self.position, self.values1, self.values2,
                                  self.last_ts, self.window)
        return res
