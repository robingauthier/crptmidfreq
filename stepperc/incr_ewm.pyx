# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, isnan
from libcpp.unordered_map cimport unordered_map
from crptmidfreq.config_loc import get_feature_folder
from cython.operator import dereference, postincrement

# Helper: compute alpha from window (half-life formula)
cdef inline double get_alpha(double window):
    """Convert half-life to alpha"""
    assert window > 0
    return 1.0 - exp(log(0.5) / window)

cdef update_ewm_values(int64_t[:] codes,
                       double[:] values,
                       int64_t[:] timestamps,
                       double alpha,
                       unordered_map[int64_t, EWMState]& state_map):
    """
    Vectorized update of EWM values.
    Uses one unordered_map that maps a code to its EWM state.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, code, ts
    cdef double value
    cdef EWMState* s
    
    for i in range(n):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Using operator[] will insert a default EWMState (with last_timestamp==0) if not present
        s = &state_map[code]  # Get pointer to the value
        
        # Check timestamp is increasing for this code
        if s.last_timestamp != 0 and ts < s.last_timestamp:
            raise ValueError("DateTime must be strictly increasing per code")

        if isnan(value):
            result[i] = np.nan
            s.last_timestamp = ts
            continue

        # Update EWM value
        if s.last_timestamp == 0:  # First occurrence
            s.last_sum = value
            s.last_wgt_sum = 1.0
        else:
            s.last_sum = s.last_sum * (1.0 - alpha) + value
            s.last_wgt_sum = s.last_wgt_sum * (1.0 - alpha) + 1.0

        # Store result for this row
        result[i] = s.last_sum / s.last_wgt_sum
        s.last_timestamp = ts

    return result

#########################################################
# Helper functions for pickling the state_map
#########################################################
# Convert state_map to a Python dict. Each entry is stored as a tuple:
# (last_sum, last_wgt_sum, last_timestamp)
cdef dict _umap_to_pydict_state(unordered_map[int64_t, EWMState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, EWMState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = ( 
                                    (<double>(dereference(it).second).last_sum),
                                    (<double>(dereference(it).second).last_wgt_sum),
                                    (<int64_t>(dereference(it).second).last_timestamp) )
        postincrement(it)
    return d

cdef unordered_map[int64_t, EWMState] _pydict_to_umap_state(dict d):
    cdef unordered_map[int64_t, EWMState] state_map
    cdef int64_t key
    cdef tuple t
    cdef EWMState s
    for key, t in d.items():
        s.last_sum = t[0]
        s.last_wgt_sum = t[1]
        s.last_timestamp = t[2]
        state_map[key] = s
    return state_map

#########################################################
# Implementation of EwmStepper
#########################################################
cdef class EwmStepper:

    def __init__(self, folder="", name="", window=1):
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        assert not isinstance(window, list)
        self.window = window
        self.alpha = get_alpha(window)
        self.state_map = unordered_map[int64_t, EWMState]()
        
    def save(self):
        import os
        import pickle

        # Construct the full path for the pickle file.
        cdef str filepath = os.path.join(self.folder, self.name + ".pkl")
        
        # Open the file in binary write mode and dump the object using pickle.
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, folder, name, window=1):
        """
        Load an instance of the class from a pickle file.

        The file is expected to be located at:
            <get_feature_folder()>/<folder>/<name>.pkl
        """
        import os
        import pickle

        # Construct the full file path
        filepath = os.path.join(get_feature_folder(), folder, name + ".pkl")
        if not os.path.exists(filepath):
            return cls(folder=folder, name=name, window=window)
        
        # Open the file and load the pickled object
        with open(filepath, "rb") as f:
            instance = pickle.load(f)

        return instance
    def __getstate__(self):
        """
        For pickling, convert state_map to a Python dict.
        """
        state = self.__dict__.copy()
        state["state_map"] = _umap_to_pydict_state(self.state_map)
        print(state)
        return state

    def __setstate__(self, state):
        print(state)
        self.__dict__.update(state)
        self.state_map = _pydict_to_umap_state(state["state_map"])

    def update(self, dt, dscode, serie):
        """
        Update EWM values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        res = update_ewm_values(codes, values, ts_array, self.alpha, self.state_map)
        return np.array(res)