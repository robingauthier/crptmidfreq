# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libcpp.unordered_map cimport unordered_map

from crptmidfreq.stepperc.incr_ewm cimport (EWMState, EwmStepper,
                                            update_ewm_values)

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Optimized Cython DetrendEwmStepper
# This is a simple extension of EwmStepper that returns the 
# difference between the original values and the EWM values
#########################################################
cdef class DetrendEwmStepper(EwmStepper):

    def update(self, dt, dscode, serie):
        """
        Update detrended EWM values. Returns original values minus EWM values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        self.validate_input(dt, dscode, serie)
        
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        
        # Calculate EWM values
        cdef np.ndarray ewm_vals = np.array(update_ewm_values(codes, values, ts_array, self.alpha, self.state_map))
        
        # Return original values minus EWM values
        return serie - ewm_vals