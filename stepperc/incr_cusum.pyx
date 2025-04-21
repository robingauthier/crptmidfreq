# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport fmax, isnan
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
cdef struct CusumState:
    double cusum_value
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Optimized update function using a combined state_map
#########################################################
cdef update_cusum_values(int64_t[:] codes,
                        double[:] values,
                        int64_t[:] timestamps,
                        double target_mean,
                        double k,
                        double h,
                        unordered_map[int64_t, CusumState]& state_map):
    """
    Vectorized update of a one-sided CUSUM statistic per code.
    Uses one unordered_map that maps a code to its state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, ts
    cdef double value, g_t, new_cusum
    cdef CusumState* s
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Using operator[] will insert a default CusumState (with last_timestamp==0) if not present
        s = &state_map[code]  # Get pointer to the value
        
        # Check timestamp is increasing for this code
        if s.last_timestamp != 0 and ts < s.last_timestamp:
            raise ValueError("DateTime must be strictly increasing per code")

        # If the series value is NaN, we skip updating CUSUM
        if isnan(value):
            result[i] = np.nan
            s.last_timestamp = ts
            continue

        # Calculate "increment" g_t: (x_t - target_mean) - k
        g_t = (value - target_mean) - k

        # One-sided CUSUM update: S_t = max(0, S_{t-1} + g_t)
        new_cusum = fmax(0.0, s.cusum_value + g_t)

        # If new_cusum > threshold, reset to zero
        if new_cusum > h:
            new_cusum = 0.0

        # Save updated cusum value for this code
        s.cusum_value = new_cusum
        s.last_timestamp = ts
        
        # Store result for this row
        result[i] = new_cusum

    return result

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (cusum_value, last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, CusumState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, CusumState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = ( 
                                      (<double>(dereference(it).second).cusum_value),
                                      (<int64_t>(dereference(it).second).last_timestamp) )
        postincrement(it)
    return d

cdef unordered_map[int64_t, CusumState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, CusumState] state_map
    cdef int64_t key
    cdef tuple t
    cdef CusumState s
    for key, t in d.items():
        s.cusum_value = t[0]
        s.last_timestamp = t[1]
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython CusumStepper
#########################################################
cdef class CusumStepper:
    cdef dict __dict__
    
    cdef public double target_mean
    cdef public double k
    cdef public double h
    cdef unordered_map[int64_t, CusumState] state_map

    def __init__(self, folder="", name="", target_mean=0.0, k=0.5, h=5.0):
        """
        Initialize CusumStepper for CUSUM detection
        
        Args:
            folder: Directory to save state (if desired)
            name: Name for saving
            target_mean: Reference mean under no-change
            k: Slack parameter (small offset from target_mean)
            h: Detection threshold
        """
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.target_mean = float(target_mean)
        self.k = float(k)
        self.h = float(h)
        self.state_map = unordered_map[int64_t, CusumState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, target_mean=0.0, k=0.5, h=5.0):
        """
        Load an instance of the class from a pickle file.
        """
        return load_instance(cls, folder, name, target_mean=target_mean, k=k, h=h)

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
        Updates the CUSUM statistic for each incoming row.
        
        Args:
            dt: np.array of datetime64 values
            dscode: np.array of int64 codes
            serie: np.array of float64 values
            
        Returns:
            np.array of float64, same length as input arrays, containing
            the updated CUSUM statistic for each row.
        """
        self.validate_input(dt, dscode, serie)
        
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        
        res = update_cusum_values(codes, values, ts_array, self.target_mean, 
                                self.k, self.h, self.state_map)
        return np.array(res)