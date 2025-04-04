
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


@njit(cache=True)
def clip_values(codes, values, timestamps, last_timestamps, low_clip=np.nan, high_clip=np.nan):
    """

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

        resloc = value
        # Store result and update last known value
        if not np.isnan(value):
            if not np.isnan(low_clip):
                resloc = max(low_clip, resloc)
            if not np.isnan(high_clip):
                resloc = min(high_clip, resloc)
        result[i] = resloc
        last_timestamps[code] = ts
    return result


class ClipStepper(BaseStepper):
    """Forward fill stepper that maintains last known values per code"""

    def __init__(self, folder='', name='', low_clip=np.nan, high_clip=np.nan):
        """
        Initialize FfillStepper for forward filling missing values

        Args:
            folder: folder for saving/loading state
            name: name for saving/loading state
        """
        super().__init__(folder, name)

        self.low_clip = low_clip
        self.high_clip = high_clip

        # Initialize empty state
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, low_clip=np.nan, high_clip=np.nan):
        """Load instance from saved state or create new if not exists"""
        return ClipStepper.load_utility(cls, folder=folder, name=name, low_clip=low_clip, high_clip=high_clip)

    def update(self, dt, dscode, serie):
        self.validate_input(dt, dscode, serie)

        # Update values and timestamps using numba function
        return clip_values(
            dscode, serie, dt.view(np.int64),
            self.last_timestamps, self.low_clip, self.high_clip
        )
