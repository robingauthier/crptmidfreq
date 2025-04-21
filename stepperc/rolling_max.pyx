# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport isnan
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from crptmidfreq.config_loc import get_feature_folder

from crptmidfreq.stepperc.rolling_base cimport (RollingState, RollingStepper,
                                                update_rolling_values)

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Update function for rolling max
#########################################################
cdef update_rolling_max(int64_t[:] timestamps,
                       int64_t[:] codes,
                       double[:] values,
                       unordered_map[int64_t, RollingState]& state_map,
                       int window):
    """
    Update rolling max values for each code.
    Uses one unordered_map that maps a code to its rolling state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, j, code
    cdef double value, current_max
    cdef RollingState* s
    cdef bint has_value
    
    # First update the rolling window for all values
    update_rolling_values(timestamps, codes, values, state_map, window)
    
    # Then compute the max for each position
    for i in range(n):
        code = codes[i]
        s = &state_map[code]
        
        # Calculate max of the rolling window
        current_max = -1e300  # A very low number to start
        has_value = False
        
        for j in range(window):
            if not isnan(s.values[j]):
                if not has_value or s.values[j] > current_max:
                    current_max = s.values[j]
                    has_value = True
        
        result[i] = current_max if has_value else np.nan
    
    return result

#########################################################
# Optimized Cython RollingMaxStepper 
#########################################################
cdef class RollingMaxStepper(RollingStepper):
    
    def update(self, dt, dscode, values):
        """
        Update rolling max values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        values: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] vals = values
        res = update_rolling_max(ts_array, codes, vals, self.state_map, self.window)
        return np.array(res)