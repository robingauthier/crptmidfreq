import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
import pickle
from config_loc import get_data_folder

## runs a stepper that calls model.predict

@njit
def update_diff_values(codes, values, timestamps, window, diff_values, last_timestamps):
    """
    Vectorized update of difference values and timestamps

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        window: window size for difference calculation
        diff_values: Dict mapping codes to current difference values
        last_timestamps: Dict mapping codes to last timestamp for each code

    Returns:
        Array of difference values corresponding to each input row
    """
    result = np.empty(len(codes), dtype=np.float64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing per code")

        # Update difference value
        old_value = diff_values.get(code, np.nan)
        if np.isnan(old_value):
            new_value = np.nan
        else:
            new_value = value - old_value

        # Store updates
        diff_values[code] = value
        last_timestamps[code] = ts

        # Store result for this row
        result[i] = new_value

    return result


class DiffStepper:
    _instances = {}  # Class variable to track loaded instances
    
    def __init__(self, folder='', name='', window=1):
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name
        self.window = window

        # Initialize empty state
        self.diff_values = Dict.empty(
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
            'diff_values': dict(self.diff_values),
            'window': self.window
        }
        print(state)
        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name, window=1):
        """Load instance from saved state or create new if not exists"""
        instance_key = f"{folder}/{name}"
        #if instance_key in cls._instances:
        #    return cls._instances[instance_key]

        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')
        
        try:
            with open(filepath, 'rb') as f:
                print(f'loading {filepath}')
                state = pickle.load(f)
            print(state)
            # Create a new instance
            instance = cls(folder=folder_path, name=name, window=state['window'])

            # Convert regular dicts back to numba Dicts
            for k, v in state['diff_values'].items():
                instance.diff_values[k] = v
            for k, v in state['last_timestamps'].items():
                instance.last_timestamps[k] = v
        except (FileNotFoundError, ValueError):
            print('Cannot load the Stepper- will create one')
            if window is None:
                raise ValueError("window parameter is required when creating new instance")
            instance = cls(folder=folder, name=name, window=window)
        
        cls._instances[instance_key] = instance
        return instance

    def update(self, dt, dscode, serie):
        """
        Update difference values for each code and return the difference values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing difference values
        """
        # Input validation
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        # Convert datetime64 to int64 nanoseconds for Numba
        timestamps = dt.astype('datetime64[ns]').astype('int64')

        # Update values and timestamps using numba function
        return update_diff_values(
            dscode, serie, timestamps,
            self.window, self.diff_values, self.last_timestamps
        )
