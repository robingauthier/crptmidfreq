import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
import pickle
from .base_stepper import BaseStepper

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

class EwmSkewStepper(BaseStepper):
    
    def __init__(self, folder='', name='', window=1):
        super().__init__(folder,name)
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
        self.save_utility()

    @classmethod
    def load(cls, folder, name, window=1):
        """Load instance from saved state or create new if not exists"""
        return EwmSkewStepper.load_utility(cls,folder=folder,name=name,window=window)

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
        self.validate_input(dt,dscode,serie)
        
        # Update values and timestamps using numba function
        return update_ewmskew_values(
            dscode, serie, dt.view(np.int64),
            self.alpha, self.ewm_values, self.ewm_squared_values,
            self.ewm_cubed_values, self.last_timestamps
        )