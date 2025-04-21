# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport isnan
from libcpp.unordered_map cimport unordered_map

from cython.operator import dereference, postincrement

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Define a C++ struct to hold all state for one code
#########################################################
cdef struct DiffState:
    double value
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Optimized update function using a combined state_map
#########################################################
cdef update_diff_values(int64_t[:] codes,
                       double[:] values,
                       int64_t[:] timestamps,
                       unordered_map[int64_t, DiffState]& state_map):
    """
    Vectorized update of difference values.
    Uses one unordered_map that maps a code to its state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, ts
    cdef double value, new_value
    cdef DiffState* s
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Using operator[] will insert a default DiffState (with last_timestamp==0) if not present
        s = &state_map[code]  # Get pointer to the value
        
        # Check timestamp is increasing for this code
        if s.last_timestamp != 0 and ts < s.last_timestamp:
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(s.last_timestamp)
            raise ValueError("DateTime must be strictly increasing per code")

        # Update difference value
        if s.last_timestamp == 0:  # First occurrence
            new_value = np.nan
        else:
            new_value = value - s.value

        # Store updates
        s.value = value
        s.last_timestamp = ts

        # Store result for this row
        result[i] = new_value

    return result

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (value, last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, DiffState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, DiffState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = ( 
                                        (<double>(dereference(it).second).value),
                                        (<int64_t>(dereference(it).second).last_timestamp) )
        postincrement(it)
    return d

cdef unordered_map[int64_t, DiffState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, DiffState] state_map
    cdef int64_t key
    cdef tuple t
    cdef DiffState s
    for key, t in d.items():
        s.value = t[0]
        s.last_timestamp = t[1]
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython DiffStepper using the combined state_map
#########################################################
cdef class DiffStepper:
    cdef dict __dict__
    
    #cdef public int window
    cdef unordered_map[int64_t, DiffState] state_map

    def __init__(self, folder="", name="", window=1):
        assert window == 1
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.window = window
        self.state_map = unordered_map[int64_t, DiffState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, window=1):
        """
        Load an instance of the class from a pickle file.
        """
        return load_instance(cls, folder, name, window=window)

    def __getstate__(self):
        """
        For pickling, convert state_map to a Python dict.
        """
        state = self.__dict__.copy()
        state["state_map"] = _umap_to_pydict_state(self.state_map)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.state_map = _pydict_to_umap_state(state["state_map"])

    def update(self, dt, dscode, serie):
        """
        Update difference values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        res = update_diff_values(codes, values, ts_array, self.state_map)
        return np.array(res)