# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
from libc.math cimport isnan
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from crptmidfreq.config_loc import get_feature_folder
from cython.operator import dereference, postincrement
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Use the typedefs from the .pxd file - don't redefine them here

cdef update_rolling_values(int64_t[:] timestamps,
                           int64_t[:] codes,
                           double[:] values,
                           unordered_map[int64_t, RollingState]& state_map,
                           int window):
    """
    Update rolling values for all codes at once.
    Uses one unordered_map that maps a code to its rolling state.
    """
    cdef int64_t n = codes.shape[0]
    cdef int64_t i, code
    
    # Loop through all values and update the state
    for i in range(n):
        update_rolling_values_loc(i, timestamps, codes, values, state_map, window)

cdef update_rolling_values_loc(
    int64_t i,
    int64_t[:] timestamps,
                           int64_t[:] codes,
                           double[:] values,
                           unordered_map[int64_t, RollingState]& state_map,
                           int window):
    """
    Update rolling values for each code.
    Uses one unordered_map that maps a code to its rolling state.
    """
    code = codes[i]
    value = values[i]
    ts = timestamps[i]

    # Using operator[] will insert a default RollingState if not present
    s = &state_map[code]  # Get pointer to the value
    
    # Check timestamp is increasing for this code
    if s.last_timestamp != 0 and ts < s.last_timestamp:
        raise ValueError("DateTime must be strictly increasing per code")

    # Initialize values vector if first occurrence 
    if s.values.size() == 0:
        s.values.resize(window, float('nan'))
        s.position = 0
        
    # Update rolling window
    position_loc = s.position
    s.values[position_loc] = value
    s.last_timestamp = ts
    s.position = (position_loc + 1) % window

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (values, position, last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, RollingState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, RollingState].iterator it = state_map.begin()
    cdef vector[double] values_vec
    cdef int i
    
    while it != state_map.end():
        values_vec = (<vector[double]>(dereference(it).second).values)
        values_array = np.empty(values_vec.size(), dtype=np.float64)
        for i in range(values_vec.size()):
            values_array[i] = values_vec[i]
            
        d[ (<int64_t>(dereference(it).first)) ] = ( 
                                        values_array,
                                        (<int64_t>(dereference(it).second).position),
                                        (<int64_t>(dereference(it).second).last_timestamp) )
        postincrement(it)
    return d

cdef unordered_map[int64_t, RollingState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, RollingState] state_map
    cdef int64_t key
    cdef tuple t
    cdef RollingState s
    cdef np.ndarray[double, ndim=1] values_array
    cdef int i
    
    for key, t in d.items():
        values_array = t[0]
        s.values.resize(values_array.shape[0])
        for i in range(values_array.shape[0]):
            s.values[i] = values_array[i]
        s.position = t[1]
        s.last_timestamp = t[2]
        state_map[key] = s
    return state_map

#########################################################
# RollingStepper implementation
#########################################################
cdef class RollingStepper:
    def __init__(self, folder="", name="", window=1):
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.window = window
        self.state_map = unordered_map[int64_t, RollingState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, window=1):
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

