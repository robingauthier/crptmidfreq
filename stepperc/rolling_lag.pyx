# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport isnan
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from crptmidfreq.config_loc import get_feature_folder

from crptmidfreq.stepperc.rolling_base cimport RollingState, RollingStepper

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Update function for rolling lag
#########################################################
cdef update_rolling_lag(int64_t[:] timestamps,
                       int64_t[:] codes,
                       double[:] values,
                       unordered_map[int64_t, RollingState]& state_map,
                       int window):
    """
    Update rolling lag values for each code.
    Unlike other rolling functions, we return the previous value at the current position
    before updating it with the new value.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, position_loc
    cdef double value, lagged_value
    cdef RollingState* s
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        
        # Using operator[] will insert a default RollingState if not present
        s = &state_map[code]
        
        # Check timestamp is increasing for this code
        if s.last_timestamp != 0 and timestamps[i] < s.last_timestamp:
            raise ValueError("DateTime must be strictly increasing per code")
            
        # Initialize values vector if first occurrence 
        if s.values.size() == 0:
            s.values.resize(window, float('nan'))
            s.position = 0
        
        position_loc = s.position
        
        # Get the lagged value before updating
        lagged_value = s.values[position_loc]
        
        # Update the current position with the new value
        s.values[position_loc] = value
        s.last_timestamp = timestamps[i]
        s.position = (position_loc + 1) % window
        
        # Return the lagged value
        result[i] = lagged_value
    
    return result

#########################################################
# Optimized Cython RollingLagStepper 
#########################################################
cdef class RollingLagStepper(RollingStepper):
    
    def update(self, dt, dscode, values):
        """
        Update rolling lag values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        values: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] vals = values
        res = update_rolling_lag(ts_array, codes, vals, self.state_map, self.window)
        return np.array(res)