import numpy as np
from numba import njit

from crptmidfreq.stepper.rolling_base import RollingStepper


@njit(cache=True)
def update_rolling_max(timestamps,
                       dscode,
                       values,
                       position,
                       rolling_dict,
                       last_timestamps,
                       window):
    """

    """
    result = np.zeros(len(dscode), dtype=np.float64)
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
            for j in range(window):
                rolling_dict[code][j] = np.nan

        position_loc = position[code]
        rolling_dict[code][position_loc] = value
        last_timestamps[code] = ts
        position[code] = (position_loc + 1) % window

        result[i] = np.max(rolling_dict[code])
    return result


class RollingMaxStepper(RollingStepper):
    def update(self, dt, dscode, values):
        self.validate_input(dt, dscode, values)
        res = update_rolling_max(dt.view(np.int64), dscode, values,
                                 self.position, self.values, self.last_ts, self.window)
        return res
