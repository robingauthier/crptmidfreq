
import numpy as np
from numba import njit, types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


def get_alpha(window=1.0):
    """Convert half-life to alpha"""
    assert window > 0
    return 1 - np.exp(np.log(0.5) / window)


@njit(cache=True)
def update_ewm_values(codes, values, timestamps, alpha, last_sum, last_wgt_sum, last_timestamps):
    """
    Vectorized update of EWM values and timestamps

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        alpha: EWM alpha parameter
        ewm_values: Dict mapping codes to current EWM values
        last_timestamps: Dict mapping codes to last timestamp for each code

    Returns:
        Array of EWM values corresponding to each input row
    """
    result = np.empty(len(codes), dtype=np.float64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        if np.isnan(value):
            result[i] = np.nan
            last_timestamps[code] = ts
            continue

        # Update EWM value
        old_last_sum = last_sum.get(code, np.nan)
        old_last_wgt_sum = last_wgt_sum.get(code, np.nan)
        if np.isnan(old_last_sum):
            last_sum[code] = value
            last_wgt_sum[code] = 1
        else:
            last_sum[code] = old_last_sum * (1 - alpha) + value
            last_wgt_sum[code] = old_last_wgt_sum * (1 - alpha) + 1

        # Store result for this row
        result[i] = last_sum[code] / last_wgt_sum[code]
        last_timestamps[code] = ts

    return result


class EwmStepper(BaseStepper):

    def __init__(self, folder='', name='', window=1):
        super().__init__(folder, name)
        assert not isinstance(window, list)
        self.window = window
        self.alpha = get_alpha(window)

        # Initialize empty state
        self.last_sum = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_wgt_sum = Dict.empty(
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
        return EwmStepper.load_utility(cls, folder=folder, name=name, window=window)

    def update(self, dt, dscode, serie):
        """
        Update EWM values for each code and return the EWM values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing EWM values
        """
        self.validate_input(dt, dscode, serie)

        # Update values and timestamps using numba function
        res = update_ewm_values(
            dscode, serie, dt.view(np.int64),
            self.alpha, self.last_sum, self.last_wgt_sum, self.last_timestamps
        )
        return res
