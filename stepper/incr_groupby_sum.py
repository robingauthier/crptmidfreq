
import numpy as np
from numba.core import types
from numba.typed import Dict, List

from crptmidfreq.stepper.base_stepper import BaseStepper

# It reduces the size of the data.. It is not a cs_sum

# @njit(cache=True)
def groupby_sum_values(dts, codes, bys, values,
                       bymap,
                       result_ts, result_by, result_sum, result_cnt
                       ):
    """
    equivalent to groupby(['by','dt']).sum() usuful for resampling the data
    """

    byg = 0

    posloc = 0
    last_ts = -1

    pos = 0
    for i in range(len(codes)):
        code = codes[i]
        by = bys[i]
        value = values[i]
        ts = dts[i]

        # Check timestamp is increasing
        if ts < last_ts:
            print(ts)
            print(code)
            print(last_ts)
            raise ValueError(f"DateTime must be strictly increasing per code {ts} last_ts={last_ts} i={i}")

        if ts > last_ts:
            bymap = Dict.empty(
                key_type=types.int64,
                value_type=types.int64
            )
            last_ts = ts
            byg = 0
            pos = len(result_by)

        if by not in bymap:
            result_ts.append(0)
            result_by.append(0)
            result_sum.append(0.0)
            result_cnt.append(0)
            bymap[by] = byg
            byg += 1

        posloc = pos+bymap[by]
        result_ts[posloc] = ts
        result_by[posloc] = by
        result_sum[posloc] += value
        result_cnt[posloc] += 1

    return result_ts, result_by, result_sum, result_cnt


class GroupbySumStepper(BaseStepper):
    """Last value // removes duplicates """

    def __init__(self, folder='', name='', is_sum=True):
        """

        """
        super().__init__(folder, name)

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
        self.is_sum = is_sum

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, is_sum=True):
        """Load instance from saved state or create new if not exists"""
        return GroupbySumStepper.load_utility(cls, folder=folder, name=name, is_sum=is_sum)

    def update(self, dt, dscode, by, serie):
        """
        """
        # Input validation
        self.validate_input(dt, dscode, serie)
        self.validate_input(dt, by, serie)

        result_ts = List.empty_list(types.int64)
        result_by = List.empty_list(types.int64)
        result_sum = List.empty_list(types.float64)
        result_cnt = List.empty_list(types.int64)

        bymap = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

        # Update values and timestamps using numba function
        result_ts, result_by, result_sum, result_cnt = groupby_sum_values(
            dt, dscode, by, serie,
            bymap,
            result_ts, result_by, result_sum, result_cnt
        )
        result_ts = np.array(result_ts, dtype=np.int64)
        result_by = np.array(result_by, dtype=np.int64)
        result_sum = np.array(result_sum, dtype=np.float64)
        result_cnt = np.array(result_cnt, dtype=np.int64)
        if self.is_sum:
            result_val = result_sum
        else:
            result_val = np.divide(
                result_sum,
                result_cnt,
                out=np.zeros_like(result_cnt),
                where=~np.isclose(result_cnt, np.zeros_like(result_cnt))
            )
        return result_ts, result_by, result_val
