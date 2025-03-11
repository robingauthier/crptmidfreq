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
def update_ewmskew_values(codes, values, timestamps, alpha, ewm_values, ewm_squared_values, 
                         ewm_cubed_values, last_timestamps):
    """
    Vectorized update of EWM values for skewness calculation

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        alpha: EWM alpha parameter
        ewm_values: Dict mapping codes to current EWM values
        ewm_squared_values: Dict mapping codes to current EWM squared values
        ewm_cubed_values: Dict mapping codes to current EWM cubed values
        last_timestamps: Dict mapping codes to last timestamp for each code

    Returns:
        Array of EWM skewness values corresponding to each input row
    """
    result = np.empty(len(codes), dtype=np.float64)
    count = np.zeros(len(codes), dtype=np.int64)  # Count per code for bias correction

    # Count occurrences of each code up to each position
    code_counts = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(len(codes)):
        code = codes[i]
        count[i] = code_counts.get(code, 0) + 1
        code_counts[code] = count[i]

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        squared_value = value * value
        cubed_value = squared_value * value
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        # Update EWM values
        old_value = ewm_values.get(code, np.nan)
        old_squared_value = ewm_squared_values.get(code, np.nan)
        old_cubed_value = ewm_cubed_values.get(code, np.nan)

        if np.isnan(old_value):
            new_value = value
            new_squared_value = squared_value
            new_cubed_value = cubed_value
            skew = 0.0  # First value has no skewness
        else:
            # Regular exponential decay
            new_value = old_value * (1 - alpha) + value * alpha
            new_squared_value = old_squared_value * (1 - alpha) + squared_value * alpha
            new_cubed_value = old_cubed_value * (1 - alpha) + cubed_value * alpha

            # Calculate skewness using EWM values
            # Skewness = (E[X^3] - 3*E[X^2]*E[X] + 2*E[X]^3) / var^(3/2)
            variance = new_squared_value - (new_value * new_value)
            if variance > 0:
                m3 = new_cubed_value - 3 * new_squared_value * new_value + \
                     2 * new_value * new_value * new_value
                skew = m3 / (variance ** 1.5)
            else:
                skew = 0.0

        # Store updates
        ewm_values[code] = new_value
        ewm_squared_values[code] = new_squared_value
        ewm_cubed_values[code] = new_cubed_value
        last_timestamps[code] = ts

        # Store result for this row
        result[i] = skew

    return result

class EwmSkewStepper:
    _instances = {}  # Class variable to track loaded instances
    
    def __init__(self, folder='', name='', window=1):
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name
        self.window = window
        self.alpha = get_alpha(window)

        # Initialize empty state
        self.ewm_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.ewm_squared_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.ewm_cubed_values = Dict.empty(
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
            'ewm_values': dict(self.ewm_values),
            'ewm_squared_values': dict(self.ewm_squared_values),
            'ewm_cubed_values': dict(self.ewm_cubed_values),
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
        
        try:
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
            for k, v in state['ewm_values'].items():
                instance.ewm_values[k] = v
            for k, v in state['ewm_squared_values'].items():
                instance.ewm_squared_values[k] = v
            for k, v in state['ewm_cubed_values'].items():
                instance.ewm_cubed_values[k] = v
            for k, v in state['last_timestamps'].items():
                instance.last_timestamps[k] = v
        except (FileNotFoundError, ValueError):
            if window is None:
                raise ValueError("window parameter is required when creating new instance")
            instance = cls(folder=folder, name=name, window=window)
        
        cls._instances[instance_key] = instance
        return instance

    def update(self, dt, dscode, serie):
        """
        Update EWM skewness values for each code and return the values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing EWM skewness values
        """
        # Input validation
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        # Convert datetime64 to int64 nanoseconds for Numba
        timestamps = dt.astype('datetime64[ns]').astype('int64')

        # Update values and timestamps using numba function
        return update_ewmskew_values(
            dscode, serie, timestamps,
            self.alpha, self.ewm_values, self.ewm_squared_values,
            self.ewm_cubed_values, self.last_timestamps
        )