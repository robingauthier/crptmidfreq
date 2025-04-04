import numpy as np
from numba import njit, types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


@njit(cache=True)
def update_cusum_values(codes, values, timestamps, target_mean, k, h,
                        cusum_dict, last_timestamps):
    """
    Vectorized update of a one-sided CUSUM statistic per code.

    Args:
        codes: array of categorical codes (int64)
        values: array of data values (float64)
        timestamps: array of timestamps (int64, e.g. from dt.view(np.int64))
        target_mean: float, the assumed 'normal' mean under no change
        k: float, slack offset to reduce false alarms
        h: float, detection threshold
        cusum_dict: Dict[int64, float64], persistent CUSUM state for each code
        last_timestamps: Dict[int64, int64], last timestamp for each code

    Returns:
        result: A float64 array of same length as inputs, containing
                the updated CUSUM values for each row.
    """
    n = len(codes)
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is strictly increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:
            raise ValueError("DateTime must be strictly increasing per code")

        # If the series value is NaN, we skip updating CUSUM
        # and store np.nan in the output. You can adapt as desired.
        if np.isnan(value):
            result[i] = np.nan
            last_timestamps[code] = ts
            continue

        # Retrieve previous cusum (default to 0.0 if not present)
        old_cusum = cusum_dict.get(code, 0.0)

        # Calculate "increment" g_t: (x_t - target_mean) - k
        g_t = (value - target_mean) - k

        # One-sided CUSUM update: S_t = max(0, S_{t-1} + g_t)
        new_cusum = max(0.0, old_cusum + g_t)

        # If new_cusum > threshold, you can:
        #  (1) reset to zero, so you detect further changes,
        #  (2) keep it so subsequent increments might detect further shifts,
        #  or (3) store an alarm somewhere else.
        if new_cusum > h:
            # For demo, we reset to 0
            new_cusum = 0.0

        # Save updated cusum value for this code
        cusum_dict[code] = new_cusum
        # Store the current timestamp
        last_timestamps[code] = ts
        # Store result for this row
        result[i] = new_cusum

    return result


class CusumStepper(BaseStepper):
    """
    One-sided CUSUM stepper that maintains a separate cumulative sum statistic
    for each code in the input data.

    On detection (cusum > threshold), it resets that code's statistic to zero.
    """

    def __init__(self, folder='', name='', target_mean=0.0, k=0.5, h=5.0):
        """
        Args:
            folder: Directory to save state (if desired)
            name: Name for saving
            target_mean: Reference mean under no-change
            k: Slack parameter (small offset from target_mean)
            h: Detection threshold
        """
        super().__init__(folder, name)

        self.target_mean = float(target_mean)
        self.k = float(k)
        self.h = float(h)

        # Dicts to store the persistent state for each code
        self.cusum_dict = Dict.empty(
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
    def load(cls, folder, name, target_mean=0.0, k=0.5, h=5.0):
        """Load instance from saved state or create new if not exists"""
        return CusumStepper.load_utility(
            cls,
            folder=folder,
            name=name,
            target_mean=target_mean,
            k=k,
            h=h
        )

    def update(self, dt, dscode, serie):
        """
        Updates the CUSUM statistic for each incoming row. Returns the
        updated CUSUM values (one per row in the input).

        Args:
            dt: np.array of datetime64 values
            dscode: np.array of int64 codes
            serie: np.array of float64 values

        Returns:
            np.array of float64, same length as input arrays, containing
            the updated CUSUM statistic for each row.
        """
        self.validate_input(dt, dscode, serie)

        # Convert datetime64 to int64 for nanosecond timestamps
        ts_int64 = dt.view(np.int64)

        # Call the numba-compiled update
        res = update_cusum_values(
            dscode,      # codes
            serie,       # values
            ts_int64,    # timestamps
            self.target_mean,
            self.k,
            self.h,
            self.cusum_dict,
            self.last_timestamps
        )
        return res
