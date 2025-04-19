# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
from libc.math cimport isnan
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Optimized groupby sum function
#########################################################
def groupby_sum_values(int64_t[:] dts, int64_t[:] codes, int64_t[:] bys, double[:] values):
    """
    Equivalent to groupby(['by','dt']).sum() useful for resampling the data
    """
    # Create unordered_map for the by groups 
    cdef unordered_map[int64_t, int64_t] bymap
    cdef int64_t byg = 0
    
    # Use vectors to store results
    cdef vector[int64_t] result_ts
    cdef vector[int64_t] result_by
    cdef vector[double] result_sum
    cdef vector[int64_t] result_cnt
    
    cdef int64_t last_ts = -1
    cdef int64_t pos = 0
    cdef int64_t posloc = 0
    cdef int64_t i, n = dts.shape[0]
    cdef int64_t code, by, ts
    cdef double value
    
    for i in range(n):
        code = codes[i]
        by = bys[i]
        value = values[i]
        ts = dts[i]
        
        # Check timestamp is increasing
        if ts < last_ts:
            print(ts)
            print(code)
            print(last_ts)
            raise ValueError(f"DateTime must be strictly increasing per code {ts} last_ts={last_ts} i={i}")
            
        if ts > last_ts:
            bymap.clear()
            last_ts = ts
            byg = 0
            pos = result_by.size()
            
        if bymap.count(by) == 0:
            result_ts.push_back(0)
            result_by.push_back(0)
            result_sum.push_back(0.0)
            result_cnt.push_back(0)
            bymap[by] = byg
            byg += 1
            
        posloc = pos + bymap[by]
        result_ts[posloc] = ts
        result_by[posloc] = by
        result_sum[posloc] += value
        result_cnt[posloc] += 1
        
    # Convert vectors to numpy arrays
    cdef int64_t size = result_ts.size()
    cdef np.ndarray[int64_t, ndim=1] np_result_ts = np.empty(size, dtype=np.int64)
    cdef np.ndarray[int64_t, ndim=1] np_result_by = np.empty(size, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] np_result_sum = np.empty(size, dtype=np.float64)
    cdef np.ndarray[int64_t, ndim=1] np_result_cnt = np.empty(size, dtype=np.int64)
    
    for i in range(size):
        np_result_ts[i] = result_ts[i]
        np_result_by[i] = result_by[i]
        np_result_sum[i] = result_sum[i]
        np_result_cnt[i] = result_cnt[i]
        
    return np_result_ts, np_result_by, np_result_sum, np_result_cnt

#########################################################
# Optimized Cython GroupbySumStepper
#########################################################
cdef class GroupbySumStepper:
    cdef dict __dict__
    cdef public bint is_sum
    
    def __init__(self, folder="", name="", is_sum=True):
        """
        Initialize GroupbySumStepper
        """
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.is_sum = is_sum

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, is_sum=True):
        """
        Load an instance of the class from a pickle file.
        """
        return load_instance(cls, folder, name, is_sum=is_sum)

    def update(self, dt, dscode, by, serie):
        """
        Perform groupby(['by','dt']).sum() on the input data
        
        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of int64 codes
            by: numpy array of int64 groupby values
            serie: numpy array of float64 values to sum
            
        Returns:
            Tuple of (timestamps, by_groups, values) for the grouped data
        """
        # Input validation
        self.validate_input(dt, dscode, serie)
        self.validate_input(dt, by, serie)
        
        # Convert datetime64 to int64
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef int64_t[:] by_values = by
        cdef double[:] values = serie
        
        # Call the groupby function
        result_ts, result_by, result_sum, result_cnt = groupby_sum_values(
            ts_array, codes, by_values, values
        )
        
        # Calculate final result value based on is_sum flag
        if self.is_sum:
            result_val = result_sum
        else:
            # Mean calculation - divide sum by count, handling zeros
            result_val = np.divide(
                result_sum,
                result_cnt,
                out=np.zeros_like(result_cnt, dtype=np.float64),
                where=~np.isclose(result_cnt, np.zeros_like(result_cnt))
            )
            
        return result_ts, result_by, result_val