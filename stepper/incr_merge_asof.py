
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper


@njit
def merge_asof_numba(
        left_timestamps, left_dscodes,
        right_timestamps, right_dscodes, right_values,
        last_right_timestamps, last_right_dscodes, last_right_values,
        memory
):
    """
    Numba-optimized merge_asof implementation with support for memory and last_right values.

    """
    ### check that right_timestamps are increacing
    ots = np.nan
    for i in range(len(right_timestamps)):
        ts = right_timestamps[i]
        if not np.isnan(ots):
            assert ts >= ots
        ots = ts
    ots = np.nan
    for i in range(len(left_timestamps)):
        ts = left_timestamps[i]
        if not np.isnan(ots):
            assert ts >= ots
        ots = ts

    n_left = len(left_timestamps)
    merged_values = np.full(n_left, np.nan, dtype=np.float64)

    # Initialize pointers and memory for each dscode
    dscode_pointers = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for dscode in np.unique(right_dscodes):
        dscode_pointers[dscode] = 0
    for dscode in np.unique(left_dscodes):
        dscode_pointers[dscode] = 0

    last_dscode_pointers = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for dscode in np.unique(last_right_dscodes):
        last_dscode_pointers[dscode] = 0
    for dscode in np.unique(left_dscodes):
        last_dscode_pointers[dscode] = 0

    for i in range(n_left):
        t_left = left_timestamps[i]
        dscode_left = left_dscodes[i]

        pointer = dscode_pointers[dscode_left]
        last_pointer = last_dscode_pointers[dscode_left]

        best_match = -1

        # Process right-side values for this dscode
        while pointer < len(right_timestamps) and right_timestamps[pointer] <= t_left:
            if right_dscodes[pointer] == dscode_left:
                best_match = pointer
                dscode_pointers[dscode_left] = pointer
            pointer += 1

        if best_match != -1:
            merged_values[i] = right_values[best_match]
        else:
            last_best_match = -1
            while last_pointer < len(last_right_timestamps) and last_right_timestamps[last_pointer] <= t_left:
                if last_right_dscodes[last_pointer] == dscode_left:
                    last_best_match = last_pointer
                    last_dscode_pointers[dscode_left] = last_pointer
                last_pointer += 1

            if last_best_match != -1:
                merged_values[i] = last_right_values[last_best_match]

    # I need last_right_values to be the last mem by dscode
    count_dscode = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for code in np.unique(right_dscodes):
        count_dscode[code] = 0

    ndscode = len(np.unique(right_dscodes))
    last_right_timestamps2 = np.empty(memory * ndscode, np.int64)
    last_right_dscodes2 = np.empty(memory * ndscode, np.int64)
    last_right_values2 = np.empty(memory * ndscode, np.float64)
    j = 0
    for i in range(len(right_timestamps)):
        i2 = len(right_timestamps) - i - 1
        code = right_dscodes[i2]
        count_dscode[code] += 1
        if count_dscode[code] < memory:
            last_right_timestamps2[j] = right_timestamps[i2]
            last_right_dscodes2[j] = right_dscodes[i2]
            last_right_values2[j] = right_values[i2]
            j += 1
    last_right_timestamps2 = last_right_timestamps2[:j][::-1]
    last_right_dscodes2 = last_right_dscodes2[:j][::-1]
    last_right_values2 = last_right_values2[:j][::-1]
    return merged_values, \
           last_right_timestamps2, \
           last_right_dscodes2, \
           last_right_values2


class MergeAsofStepper(BaseStepper):
    def __init__(self, folder='', name='', p=10):
        """
        Initialize MergeAsofStepper for incremental merge_asof operations

        Args:
            folder: Folder for saving/loading state
            name: Name for saving/loading state
            direction: Merge direction ("backward", "forward", or "nearest")
            p: Number of most recent values to keep in memory per dscode
        """
        super().__init__(folder,name)
        self.p = p  # Number of recent entries to retain

        # Store last right-side data
        self.right_timestamps = np.array([], dtype=np.int64)
        self.right_dscodes = np.array([], dtype=np.int64)
        self.right_values = np.array([], dtype=np.float64)
    
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return MergeAsofStepper.load_utility(cls,folder=folder,name=name)


    def update(self, left_timestamps, left_dscodes, right_timestamps, right_dscodes, right_values):
        """
        Update merge state with new data and return merged values

        Args:
            left_timestamps: Array of timestamps to merge into
            left_dscodes: Array of codes for left side
            left_values: Array of values from left side
            right_timestamps: Array of timestamps to merge from
            right_dscodes: Array of codes for right side
            right_values: Array of values to merge

        Returns:
            Tuple of (merged_timestamps, merged_dscodes, merged_left_values, merged_right_values)
        """
        # Input validation
        self.validate_input(left_timestamps,left_dscodes,np.zeros_like(left_dscodes))
        self.validate_input(right_timestamps,right_dscodes,right_values)
        
        # Input validation and type conversion
        left_timestamps = self._to_int64(left_timestamps)
        right_timestamps = self._to_int64(right_timestamps)

        # Perform merge using Numba-accelerated function
        merged_values, \
        last_right_timestamps2, \
        last_right_dscodes2, \
        last_right_values2 = merge_asof_numba(
            left_timestamps, left_dscodes,
            right_timestamps, right_dscodes, right_values,
            self.right_timestamps, self.right_dscodes, self.right_values,
            self.p
        )
        self.right_timestamps = last_right_timestamps2
        self.right_dscodes = last_right_dscodes2
        self.right_values = last_right_values2
        return merged_values

    def _to_int64(self, timestamps):
        """Convert timestamps to int64 if they are datetime64"""
        if timestamps.dtype.kind == 'M':
            return timestamps.astype('datetime64[ns]').astype('int64')
        return timestamps
