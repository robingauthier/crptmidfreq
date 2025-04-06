# distutils: language = c++

import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Define a C++ struct to hold all state for one code
#########################################################
cdef struct EWMState:
    double last_sum
    double last_wgt_sum
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Export update function for use by other modules
#########################################################
cdef update_ewm_values(int64_t[:] codes,
                      double[:] values,
                      int64_t[:] timestamps,
                      double alpha,
                      unordered_map[int64_t, EWMState]& state_map)

#########################################################
# Export EwmStepper class
#########################################################
cdef class EwmStepper:
    cdef dict __dict__
    # we should not declare these as public. Otherwise they are not part of the __dict__
    #cdef public double window
    #cdef public double alpha
    cdef unordered_map[int64_t, EWMState] state_map