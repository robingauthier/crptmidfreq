# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
from libc.math cimport isnan, fmax, fmin
from libcpp.unordered_map cimport unordered_map
from crptmidfreq.config_loc import get_feature_folder
from cython.operator import dereference, postincrement
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Define a C++ struct to hold all state for one code
#########################################################
cdef struct ClipState:
    int64_t last_timestamp  # Sentinel: 0 indicates uninitialized

#########################################################
# Optimized update function using a combined state_map
#########################################################
cdef clip_values(int64_t[:] codes,
                double[:] values,
                int64_t[:] timestamps,
                unordered_map[int64_t, ClipState]& state_map,
                double low_clip, 
                double high_clip):
    """
    Vectorized clip operation.
    Uses one unordered_map that maps a code to its state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, ts
    cdef double value, resloc
    cdef ClipState* s
    cdef bint low_valid = not isnan(low_clip)
    cdef bint high_valid = not isnan(high_clip)
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Using operator[] will insert a default ClipState (with last_timestamp==0) if not present
        s = &state_map[code]  # Get pointer to the value
        
        # Check timestamp is increasing for this code
        if s.last_timestamp != 0 and ts < s.last_timestamp:
            raise ValueError("DateTime must be strictly increasing per code")

        resloc = value
        # Store result and update last known value
        if not isnan(value):
            if low_valid:
                resloc = fmax(low_clip, resloc)
            if high_valid:
                resloc = fmin(high_clip, resloc)

        result[i] = resloc
        s.last_timestamp = ts

    return result

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, ClipState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, ClipState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = (<int64_t>(dereference(it).second).last_timestamp)
        postincrement(it)
    return d

cdef unordered_map[int64_t, ClipState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, ClipState] state_map
    cdef int64_t key
    cdef int64_t ts
    cdef ClipState s
    for key, ts in d.items():
        s.last_timestamp = ts
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython ClipStepper using the combined state_map
#########################################################
cdef class ClipStepper:
    cdef dict __dict__
    
    cdef public double low_clip
    cdef public double high_clip
    cdef unordered_map[int64_t, ClipState] state_map

    def __init__(self, folder="", name="", low_clip=float("nan"), high_clip=float("nan")):
        """
        Initialize ClipStepper for clipping values
        """
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.state_map = unordered_map[int64_t, ClipState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, low_clip=float("nan"), high_clip=float("nan")):
        """
        Load an instance of the class from a pickle file.
        """
        return load_instance(cls, folder, name, low_clip=low_clip, high_clip=high_clip)

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
        Clip values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        res = clip_values(codes, values, ts_array, self.state_map, self.low_clip, self.high_clip)
        return np.array(res)