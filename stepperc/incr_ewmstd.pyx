# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, isnan, sqrt
from libcpp.unordered_map cimport unordered_map
from crptmidfreq.config_loc import get_feature_folder
from cython.operator import dereference, postincrement

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

# Helper: compute alpha from window (half-life formula)
cdef inline double get_alpha(double window):
    """Convert half-life to alpha"""
    assert window > 0
    return 1.0 - exp(log(0.5) / window)

#########################################################
# Define a C++ struct to hold all state for one code
#########################################################
cdef struct EWMState:
    double last_sum
    double last_wgt_sum
    double last_sum_sq
    double last_wgt_sum_sq
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Optimized update function using a combined state_map
#########################################################
cdef update_ewmstd_values(int64_t[:] codes,
                         double[:] values,
                         int64_t[:] timestamps,
                         double alpha,
                         unordered_map[int64_t, EWMState]& state_map):
    """
    Vectorized update of EWM values for standard deviation calculation.
    Uses one unordered_map that maps a code to its EWM state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, ts
    cdef double value, squared_value
    cdef double mean, mean_sq, variance, std
    cdef EWMState* s
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        squared_value = value * value
        ts = timestamps[i]

        # Using operator[] will insert a default EWMState (with last_timestamp==0) if not present
        s = &state_map[code]  # Get pointer to the value
        
        # Check timestamp is increasing for this code
        if s.last_timestamp != 0 and ts < s.last_timestamp:
            raise ValueError("DateTime must be strictly increasing per code")

        # Update EWM values for both value and squared value
        if s.last_timestamp == 0:  # First occurrence
            # First value
            s.last_sum = value
            s.last_wgt_sum = 1.0
            s.last_sum_sq = squared_value
            s.last_wgt_sum_sq = 1.0
            std = 0.0  # First value has no standard deviation
        else:
            # Update sums and weight sums
            s.last_sum = s.last_sum * (1.0 - alpha) + value
            s.last_wgt_sum = s.last_wgt_sum * (1.0 - alpha) + 1.0
            s.last_sum_sq = s.last_sum_sq * (1.0 - alpha) + squared_value
            s.last_wgt_sum_sq = s.last_wgt_sum_sq * (1.0 - alpha) + 1.0

            # Calculate mean and mean of squares
            mean = s.last_sum / s.last_wgt_sum
            mean_sq = s.last_sum_sq / s.last_wgt_sum_sq

            # Calculate variance and std
            variance = mean_sq - (mean * mean)
            std = sqrt(variance) if variance > 0 else 0.0

        # Store result for this row
        result[i] = std
        s.last_timestamp = ts

    return result

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (last_sum, last_wgt_sum, last_sum_sq, last_wgt_sum_sq, last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, EWMState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, EWMState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = ( 
                                        (<double>(dereference(it).second).last_sum),
                                        (<double>(dereference(it).second).last_wgt_sum),
                                        (<double>(dereference(it).second).last_sum_sq),
                                        (<double>(dereference(it).second).last_wgt_sum_sq),
                                        (<int64_t>(dereference(it).second).last_timestamp) )
        postincrement(it)
    return d

cdef unordered_map[int64_t, EWMState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, EWMState] state_map
    cdef int64_t key
    cdef tuple t
    cdef EWMState s
    for key, t in d.items():
        s.last_sum = t[0]
        s.last_wgt_sum = t[1]
        s.last_sum_sq = t[2]
        s.last_wgt_sum_sq = t[3]
        s.last_timestamp = t[4]
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython EwmStdStepper using the combined state_map
#########################################################
cdef class EwmStdStepper:
    cdef dict __dict__
    
    cdef public double window
    cdef public double alpha
    cdef unordered_map[int64_t, EWMState] state_map

    def __init__(self, folder="", name="", window=1):
        assert window > 0
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.window = window
        self.alpha = get_alpha(window)
        self.state_map = unordered_map[int64_t, EWMState]()

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, window=1):
        return cls.load_utility(cls, folder=folder, name=name, window=window)

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
        Update EWM standard deviation values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        res = update_ewmstd_values(codes, values, ts_array, self.alpha, self.state_map)
        return np.array(res)