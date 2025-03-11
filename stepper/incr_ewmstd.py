import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
import pickle
from config_loc import get_data_folder

@njit
def get_alpha(window):
    """Convert half-life to alpha"""
    return 1 - np.exp(np.log(0.5) / window)

@njit
def update_ewmstd_values(codes, values, timestamps, alpha, last_sum, last_wgt_sum, last_sum_sq, last_wgt_sum_sq, last_timestamps):
    """
    Vectorized update of EWM values and squared values for standard deviation calculation

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        alpha: EWM alpha parameter
        last_sum: Dict mapping codes to current sum of values
        last_wgt_sum: Dict mapping codes to current sum of weights
        last_sum_sq: Dict mapping codes to current sum of squared values
        last_wgt_sum_sq: Dict mapping codes to current sum of weights for squared values
        last_timestamps: Dict mapping codes to last timestamp for each code

    Returns:
        Array of EWM standard deviation values corresponding to each input row
    """
    result = np.empty(len(codes), dtype=np.float64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        squared_value = value * value
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        # Update EWM values for both value and squared value
        old_last_sum = last_sum.get(code, np.nan)
        old_last_wgt_sum = last_wgt_sum.get(code, np.nan)
        old_last_sum_sq = last_sum_sq.get(code, np.nan)
        old_last_wgt_sum_sq = last_wgt_sum_sq.get(code, np.nan)

        if np.isnan(old_last_sum):
            # First value
            last_sum[code] = value
            last_wgt_sum[code] = 1
            last_sum_sq[code] = squared_value
            last_wgt_sum_sq[code] = 1
            std = 0.0  # First value has no standard deviation
        else:
            # Update sums and weight sums
            last_sum[code] = old_last_sum * (1 - alpha) + value
            last_wgt_sum[code] = old_last_wgt_sum * (1 - alpha) + 1
            last_sum_sq[code] = old_last_sum_sq * (1 - alpha) + squared_value
            last_wgt_sum_sq[code] = old_last_wgt_sum_sq * (1 - alpha) + 1

            # Calculate mean and mean of squares
            mean = last_sum[code] / last_wgt_sum[code]
            mean_sq = last_sum_sq[code] / last_wgt_sum_sq[code]

            # Calculate variance and std
            variance = mean_sq - (mean * mean)
            std = np.sqrt(variance) if variance > 0 else 0.0

        # Store result for this row
        result[i] = std
        last_timestamps[code] = ts

    return result

class EwmStdStepper:
    _instances = {}  # Class variable to track loaded instances
    
    def __init__(self, folder='', name='', window=1):
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name
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
        self.last_sum_sq = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_wgt_sum_sq = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        """Save internal state to file"""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        state = {
            'last_timestamps': dict(self.last_timestamps),
            'last_sum': dict(self.last_sum),
            'last_wgt_sum': dict(self.last_wgt_sum),
            'last_sum_sq': dict(self.last_sum_sq),
            'last_wgt_sum_sq': dict(self.last_wgt_sum_sq),
            'window': self.window,
            'alpha': self.alpha
        }
        print(state)
        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name, window=None):
        """Load instance from saved state or create new if not exists"""
        instance_key = f"{folder}/{name}"
        #if instance_key in cls._instances:
        #    return cls._instances[instance_key]
            
        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            instance = cls(folder=folder, name=name, window=window)
            return instance


        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(state)
        # Create a new instance
        instance = cls(
            folder=folder_path,
            name=name,
            window=state['window']
        )
        instance.alpha = state['alpha']

        # Convert regular dicts back to numba Dicts
        for k, v in state['last_sum'].items():
            instance.last_sum[k] = v
        for k, v in state['last_wgt_sum'].items():
            instance.last_wgt_sum[k] = v
        for k, v in state['last_sum_sq'].items():
            instance.last_sum_sq[k] = v
        for k, v in state['last_wgt_sum_sq'].items():
            instance.last_wgt_sum_sq[k] = v
        for k, v in state['last_timestamps'].items():
            instance.last_timestamps[k] = v

        cls._instances[instance_key] = instance
        return instance

    def update(self, dt, dscode, serie):
        """
        Update EWM standard deviation values for each code and return the values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing EWM standard deviation values
        """
        # Input validation
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        # Convert datetime64 to int64 nanoseconds for Numba
        timestamps = dt.astype('datetime64[ns]').astype('int64')

        # Update values and timestamps using numba function
        return update_ewmstd_values(
            dscode, serie, timestamps,
            self.alpha, self.last_sum, self.last_wgt_sum,
            self.last_sum_sq, self.last_wgt_sum_sq, self.last_timestamps
        )