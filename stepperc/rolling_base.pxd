# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Define a C++ struct to hold all state for one code
#########################################################
cdef struct RollingState:
    vector[double] values
    int64_t position
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Base update function using a combined state_map
#########################################################
cdef update_rolling_values(int64_t[:] timestamps,
                           int64_t[:] codes,
                           double[:] values,
                           unordered_map[int64_t, RollingState]& state_map,
                           int window)

#########################################################
# Optimized Cython RollingStepper using the combined state_map
#########################################################
cdef class RollingStepper:
    cdef dict __dict__
    
    cdef public int window
    cdef unordered_map[int64_t, RollingState] state_map