# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport exp, log
from libcpp.unordered_map cimport unordered_map

from cython.operator import dereference, postincrement

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

import numpy as np


# Helper: compute alpha from window (half-life formula)
cdef inline double get_alpha(int window):
    return 1.0 - exp(log(0.5) / window)

#########################################################
# Define a C++ struct to hold all state for one code
#########################################################
cdef struct EWMState:
    double value
    double squared_value
    double cubed_value
    double fourth_value
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Optimized update function using a combined state_map
#########################################################
cdef update_ewmkurt_values(int64_t[:] codes,
                           double[:] values,
                           int64_t[:] timestamps,
                           double alpha,
                           unordered_map[int64_t, EWMState]& state_map):
    """
    Vectorized update of EWM values for kurtosis calculation.
    Uses one unordered_map that maps a code to its EWM state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, ts
    cdef double value, squared_value, cubed_value, fourth_value
    cdef double new_value, new_squared_value, new_cubed_value, new_fourth_value
    cdef double variance, m4, kurt
    cdef EWMState* s # we need to use references to prevent copying the struct
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        squared_value = value * value
        cubed_value = squared_value * value
        fourth_value = squared_value * squared_value
        ts = timestamps[i]

        # Using operator[] will insert a default EWMState (with last_timestamp==0) if not present.
        #s = state_map[code]    
        s = &state_map[code]  # Get pointer to the value

        if s.last_timestamp == 0:
            # First occurrence: initialize state with current values.
            s.value = value
            s.squared_value = squared_value
            s.cubed_value = cubed_value
            s.fourth_value = fourth_value
            s.last_timestamp = ts
            kurt = 0.0
        else:
            if ts < s.last_timestamp:
                raise ValueError("DateTime must be strictly increasing per code")
            new_value       = s.value       * (1.0 - alpha) + value          * alpha
            new_squared_value = s.squared_value * (1.0 - alpha) + squared_value * alpha
            new_cubed_value   = s.cubed_value   * (1.0 - alpha) + cubed_value   * alpha
            new_fourth_value  = s.fourth_value  * (1.0 - alpha) + fourth_value  * alpha

            variance = new_squared_value - new_value * new_value
            if variance > 1e-14:
                m4 = new_fourth_value - 4.0 * new_cubed_value * new_value \
                     + 6.0 * new_squared_value * new_value * new_value \
                     - 3.0 * new_value**4
                kurt = m4 / (variance * variance) - 3.0
            else:
                kurt = 0.0

            s.value         = new_value
            s.squared_value = new_squared_value
            s.cubed_value   = new_cubed_value
            s.fourth_value  = new_fourth_value
            s.last_timestamp = ts
        #state_map[code] = s  # Save updated state.
        result[i] = kurt
    return result

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (value, squared_value, cubed_value, fourth_value, last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, EWMState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, EWMState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = ( (<double>(dereference(it).second).value),
                                           (<double>(dereference(it).second).squared_value),
                                           (<double>(dereference(it).second).cubed_value),
                                           (<double>(dereference(it).second).fourth_value),
                                           (<int64_t>(dereference(it).second).last_timestamp) )
        postincrement(it)
    return d

cdef unordered_map[int64_t, EWMState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, EWMState] state_map
    cdef int64_t key
    cdef tuple t
    cdef EWMState s
    for key, t in d.items():
        s.value = t[0]
        s.squared_value = t[1]
        s.cubed_value = t[2]
        s.fourth_value = t[3]
        s.last_timestamp = t[4]
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython EwmKurtStepper using the combined state_map
#########################################################
cdef class EwmKurtStepper:
    cdef dict __dict__
    
    cdef public int window
    cdef public double alpha
    cdef unordered_map[int64_t, EWMState] state_map

    def __init__(self, folder="", name="", window=1):
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.window = window
        self.alpha = get_alpha(window)
        self.state_map = unordered_map[int64_t, EWMState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, int window=1):
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
        Update EWM kurtosis values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        res= update_ewmkurt_values(codes, values, ts_array, self.alpha, self.state_map)
        return np.array(res)
