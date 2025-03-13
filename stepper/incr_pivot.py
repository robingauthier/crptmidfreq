import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict


@njit
def incremental_pivot(dt, dscode, values):
    """
    Incremental pivot function that organizes data by unique timestamps and 
    assigns each dscode as a separate column without using pandas.

    Args:
        dt (np.ndarray): Array of datetime64 (int64 nanoseconds).
        dscode (np.ndarray): Array of asset codes (int64 or str).
        values (np.ndarray): Array of values (float64).

    Returns:
        tuple: (timestamps, pivot_dict) where:
            - timestamps (np.ndarray): Unique sorted timestamps
            - pivot_dict (dict): {dscode: np.ndarray of values aligned to timestamps}
    """

    # Get unique timestamps in sorted order
    unique_times = np.unique(dt)

    # Get unique dscode values
    unique_codes = np.unique(dscode)

    # Create output dictionary where each dscode has a zero-filled array
    pivot_dict = {code: np.full(len(unique_times), np.nan, dtype=np.float64) for code in unique_codes}

    # Create a mapping from timestamp to row index
    time_to_index = {unique_times[i]: i for i in range(len(unique_times))}

    # Fill pivot_dict by mapping each (timestamp, dscode) to the corresponding value
    for i in range(len(dt)):
        row_idx = time_to_index[dt[i]]
        pivot_dict[dscode[i]][row_idx] = values[i]

    return unique_times, pivot_dict



class PivotStepper:
    """Like .pivot_table in pandas"""

    def __init__(self, folder='', name=''):
        """
        Initialize FfillStepper for forward filling missing values

        Args:
            folder: folder for saving/loading state
            name: name for saving/loading state
        """
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name

        # Initialize empty state
        self.last_values = Dict.empty(
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
            'last_values': dict(self.last_values)
        }

        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state"""
        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            return cls(folder=folder, name=name)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create new instance
        instance = cls(folder=folder, name=name)

        # Convert regular dicts back to numba Dicts
        for k, v in state['last_values'].items():
            instance.last_values[k] = v
        for k, v in state['last_timestamps'].items():
            instance.last_timestamps[k] = v

        return instance

    def update(self, dt, dscode, serie):
        """
        Update forward fill state and return filled values

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing forward-filled values
        """
        # Input validation
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        # Convert datetime64 to int64 nanoseconds for Numba
        timestamps = dt.astype('datetime64[ns]').astype('int64')

        # Update values and timestamps using numba function
        return ffill_values(
            dscode, serie, timestamps,
            self.last_values, self.last_timestamps
        )
