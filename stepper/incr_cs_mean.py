import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
from crptmidfreq.stepper.base_stepper import BaseStepper


@njit(cache=True)
def update_cs_mean_values(codes, values, bys, wgts, timestamps,
                          last_timestamps, last_sums, last_wgts,
                          is_sum):
    """
    """
    result = np.empty(len(codes), dtype=np.float64)

    g_last_ts = 0
    for k, v in last_timestamps.items():
        g_last_ts = max(v, g_last_ts)

    # j is a second iterator
    j = 0
    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        by = bys[i]
        wgt = wgts[i]
        ts = timestamps[i]

        if by not in last_sums:
            last_sums[by] = 0.0
            last_wgts[by] = 0.0

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Important to have strictly increasing
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing per code")

        # Check timestamp is increasing accross dscode
        if ts < g_last_ts:  # Important to have strictly increasing
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing accross instruments")

        if ts != g_last_ts:
            # we must not fill for j=i
            while j < i:
                # Store result for this row
                by2 = bys[j]
                if last_wgts[by2] > 0:
                    result[j] = last_sums[by2]/last_wgts[by2]
                else:
                    result[j] = 0.0
                j += 1

            # reset everything
            for k, v in last_sums.items():
                last_sums[k] = 0
                last_wgts[k] = 0

        # Store updates
        last_timestamps[code] = ts
        g_last_ts = ts
        last_sums[by] += value*wgt
        last_wgts[by] += wgt

    while j < len(codes):
        # the last value is not assigned
        by2 = bys[j]
        if is_sum:
            result[j] = last_sums[by2]
        elif last_wgts[by2] > 0:
            result[j] = last_sums[by2]/last_wgts[by2]
        else:
            result[j] = 0.0
        j = j+1
    return result


class csMeanStepper(BaseStepper):

    def __init__(self, folder='', name='', is_sum=False):
        super().__init__(folder, name)

        # Initialize empty state
        self.last_sums = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_wgts = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.is_sum = is_sum

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, is_sum=False):
        """Load instance from saved state or create new if not exists"""
        return csMeanStepper.load_utility(cls, folder=folder, name=name, is_sum=is_sum)

    def update(self, dt, dscode, serie, by=None, wgt=None):
        """
        Update difference values for each code and return the difference values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process
            by : numpy array of by values ( all, sector, cluster)
            wgt : numpy array of weight/univ flag [0-1]

        Returns:
            numpy array of same length as input arrays containing difference values
        """
        if wgt is None:
            wgt = np.ones_like(serie)
        if by is None:
            by = np.ones_like(serie)
        self.validate_input(dt, dscode, serie, by=by, wgt=wgt)

        # by must be integers
        by = by.astype('int64')

        # Update values and timestamps using numba function
        return update_cs_mean_values(
            dscode, serie, by, wgt, dt.view(np.int64),
            self.last_timestamps, self.last_sums, self.last_wgts,
            self.is_sum
        )
