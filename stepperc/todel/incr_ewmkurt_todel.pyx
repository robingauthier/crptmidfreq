# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport exp, log
from libcpp.algorithm cimport sort
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cython.operator import dereference, postincrement

from crptmidfreq.config_loc import get_feature_folder

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

# Helper: compute alpha from window (half-life formula)
cdef inline double get_alpha(int window):
    return 1.0 - exp(log(0.5) / window)

#########################################################
# Improved struct to hold state for EWM calculations
#########################################################
cdef struct EWMState:
    double value           # EWM of the values
    double squared_value   # EWM of the squared values
    double cubed_value     # EWM of the cubed values
    double fourth_value    # EWM of the fourth values
    int64_t last_timestamp # Sentinel: 0 indicates uninitialized

#########################################################
# Optimized state storage using sorted vector instead of unordered_map
#########################################################
cdef class StateVector:
    cdef vector[pair[int64_t, EWMState]] data
    cdef bint is_sorted
    
    def __cinit__(self):
        self.is_sorted = False
    
    cdef EWMState* get_state_ptr(self, int64_t code):
        # If not sorted, binary search won't work, so sort first
        if not self.is_sorted:
            sort(self.data.begin(), self.data.end())
            self.is_sorted = True
        
        # Binary search for the code
        cdef int low = 0
        cdef int high = self.data.size() - 1
        cdef int mid
        
        while low <= high:
            mid = (low + high) // 2
            if self.data[mid].first == code:
                return &self.data[mid].second
            elif self.data[mid].first < code:
                low = mid + 1
            else:
                high = mid - 1
        
        # Not found, insert new entry
        cdef EWMState new_state
        new_state.last_timestamp = 0  # Mark as uninitialized
        self.data.push_back(pair[int64_t, EWMState](code, new_state))
        self.is_sorted = False  # Mark as needing resort
        
        # Return pointer to the newly inserted state
        return &self.data[self.data.size() - 1].second
    
    cdef dict to_dict(self):
        cdef dict d = {}
        cdef size_t i
        for i in range(self.data.size()):
            d[self.data[i].first] = (
                self.data[i].second.value,
                self.data[i].second.squared_value,
                self.data[i].second.cubed_value,
                self.data[i].second.fourth_value,
                self.data[i].second.last_timestamp
            )
        return d
    
    @staticmethod
    cdef StateVector from_dict(dict d):
        cdef StateVector sv = StateVector()
        cdef int64_t key
        cdef tuple t
        cdef EWMState s
        for key, t in d.items():
            s.value = t[0]
            s.squared_value = t[1]
            s.cubed_value = t[2]
            s.fourth_value = t[3]
            s.last_timestamp = t[4]
            sv.data.push_back(pair[int64_t, EWMState](key, s))
        return sv

#########################################################
# Optimized grouping and batch processing
#########################################################
cpdef np.ndarray[double, ndim=1] update_ewmkurt_values_optimized(
        np.ndarray[int64_t, ndim=1] codes_array,
        np.ndarray[double, ndim=1] values_array,
        np.ndarray[int64_t, ndim=1] timestamps_array,
        double alpha,
        StateVector state_vector):
    """
    Optimized EWM values update for kurtosis calculation using batch processing.
    """
    cdef int64_t n = codes_array.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.empty(n, dtype=np.float64)
    
    # Group indices by code
    cdef dict code_indices = {}
    cdef int64_t i, code
    
    for i in range(n):
        code = codes_array[i]
        if code not in code_indices:
            code_indices[code] = []
        code_indices[code].append(i)
    
    # Process each code group
    cdef list indices
    cdef int64_t idx, ts
    cdef double value, squared_value, cubed_value, fourth_value
    cdef double variance, m4, kurt
    cdef EWMState* state_ptr
    
    for code, indices in code_indices.items():
        # Sort indices by timestamp
        indices.sort(key=lambda idx: timestamps_array[idx])
        
        # Process indices for this code in timestamp order
        for idx in indices:
            value = values_array[idx]
            squared_value = value * value
            cubed_value = squared_value * value
            fourth_value = squared_value * squared_value
            ts = timestamps_array[idx]
            
            # Get state pointer for this code
            state_ptr = state_vector.get_state_ptr(code)
            
            if state_ptr.last_timestamp == 0:
                # First occurrence: initialize state with current values
                state_ptr.value = value
                state_ptr.squared_value = squared_value
                state_ptr.cubed_value = cubed_value
                state_ptr.fourth_value = fourth_value
                state_ptr.last_timestamp = ts
                result[idx] = 0.0
            else:
                if ts < state_ptr.last_timestamp:
                    raise ValueError("DateTime must be strictly increasing per code")
                
                # Update EWM statistics
                state_ptr.value = state_ptr.value * (1.0 - alpha) + value * alpha
                state_ptr.squared_value = state_ptr.squared_value * (1.0 - alpha) + squared_value * alpha
                state_ptr.cubed_value = state_ptr.cubed_value * (1.0 - alpha) + cubed_value * alpha
                state_ptr.fourth_value = state_ptr.fourth_value * (1.0 - alpha) + fourth_value * alpha
                
                # Calculate kurtosis
                variance = state_ptr.squared_value - state_ptr.value * state_ptr.value
                if variance > 1e-14:
                    m4 = state_ptr.fourth_value - 4.0 * state_ptr.cubed_value * state_ptr.value \
                         + 6.0 * state_ptr.squared_value * state_ptr.value * state_ptr.value \
                         - 3.0 * state_ptr.value**4
                    kurt = m4 / (variance * variance) - 3.0
                else:
                    kurt = 0.0
                
                state_ptr.last_timestamp = ts
                result[idx] = kurt
    
    return result

#########################################################
# Multi-threaded optimization using thread pools
#########################################################
def process_in_chunks(np.ndarray[int64_t, ndim=1] codes_array,
                      np.ndarray[double, ndim=1] values_array,
                      np.ndarray[int64_t, ndim=1] timestamps_array,
                      double alpha,
                      StateVector state_vector,
                      int num_chunks=8):
    """
    Process the data in chunks for multi-threading without OpenMP.
    
    Instead of using OpenMP, you can use this with Python's concurrent.futures:
    
    from concurrent.futures import ThreadPoolExecutor
    
    def update_parallel(codes, values, timestamps, alpha, state_vector, num_threads=8):
        chunk_size = len(codes) // num_threads
        futures = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                start = i * chunk_size
                end = None if i == num_threads - 1 else (i + 1) * chunk_size
                futures.append(
                    executor.submit(
                        process_in_chunks,
                        codes[start:end],
                        values[start:end],
                        timestamps[start:end],
                        alpha,
                        state_vector,
                        1  # single chunk since we're already splitting
                    )
                )
        
        # Combine results
        results = []
        for future in futures:
            results.append(future.result())
        return np.concatenate(results)
    """
    cdef int chunk_size = codes_array.shape[0] // num_chunks
    cdef list results = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = None if i == num_chunks - 1 else (i + 1) * chunk_size
        
        chunk_codes = codes_array[start:end]
        chunk_values = values_array[start:end]
        chunk_timestamps = timestamps_array[start:end]
        
        chunk_result = update_ewmkurt_values_optimized(
            chunk_codes, chunk_values, chunk_timestamps, alpha, state_vector
        )
        results.append(chunk_result)
    
    return np.concatenate(results)

#########################################################
# Optimized EwmKurtStepper class
#########################################################
cdef class EwmKurtStepper:
    cdef dict __dict__
    
    cdef public int window
    cdef public double alpha
    cdef StateVector state_vector

    def __init__(self, folder="", name="", window=1):
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.window = window
        self.alpha = get_alpha(window)
        self.state_vector = StateVector()

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, int window=1):
        return cls.load_utility(cls, folder=folder, name=name, window=window)

    def __getstate__(self):
        """
        For pickling, convert state_vector to a Python dict.
        """
        state = self.__dict__.copy()
        state["state_vector"] = self.state_vector.to_dict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.state_vector = StateVector.from_dict(state["state_vector"])

    def update(self, dt, dscode, serie):
        """
        Update EWM kurtosis values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef np.ndarray[int64_t, ndim=1] codes = np.asarray(dscode, dtype=np.int64)
        cdef np.ndarray[double, ndim=1] values = np.asarray(serie, dtype=np.float64)
        
        return update_ewmkurt_values_optimized(codes, values, ts_array, self.alpha, self.state_vector)