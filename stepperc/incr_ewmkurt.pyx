# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import os
import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from libcpp.unordered_map cimport unordered_map
from cython.operator import dereference, postincrement
from crptmidfreq.config_loc import get_feature_folder


# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

# Helper: compute alpha from window (half-life formula)
cdef inline double get_alpha(int window):
    return 1.0 - exp(log(0.5) / window)

#########################################################
# Update function using memoryviews and unordered_maps
#########################################################
cdef update_ewmkurt_values(int64_t[:] codes,
                          double[:] values,
                          int64_t[:] timestamps,
                          double alpha,
                          unordered_map[int64_t, double]& ewm_values,
                          unordered_map[int64_t, double]& ewm_squared_values,
                          unordered_map[int64_t, double]& ewm_cubed_values,
                          unordered_map[int64_t, double]& ewm_fourth_values,
                          unordered_map[int64_t, int64_t]& last_timestamps):
    """
    Vectorized update of EWM values for kurtosis calculation.
    
    Parameters:
      codes: 1D memory view of int64_t codes.
      values: 1D memory view of float64 values.
      timestamps: 1D memory view of int64_t timestamps.
      alpha: smoothing parameter.
      The remaining parameters are C++ unordered_maps mapping a code (int64_t) to the current EWM state.
      
    Returns:
      A NumPy array (float64) with the updated EWM kurtosis values for each row.
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i
    cdef int64_t code, ts, last_ts
    cdef double value, squared_value, cubed_value, fourth_value
    cdef double old_value, old_squared_value, old_cubed_value, old_fourth_value
    cdef double new_value, new_squared_value, new_cubed_value, new_fourth_value
    cdef double variance, m4, kurt
    cdef double nan = np.nan  # Use numpy.nan

    for i in range(n):
        code = codes[i]
        value = values[i]
        squared_value = value * value
        cubed_value = squared_value * value
        fourth_value = squared_value * squared_value
        ts = timestamps[i]

        # Get last timestamp (default to 0 if not present)
        if last_timestamps.find(code) == last_timestamps.end():
            last_ts = 0
        else:
            last_ts = last_timestamps[code]

        if ts < last_ts:
            raise ValueError("DateTime must be strictly increasing per code")

        # Retrieve previous EWM values; if missing, default to nan
        if ewm_values.find(code) == ewm_values.end():
            old_value = nan
        else:
            old_value = ewm_values[code]
        if ewm_squared_values.find(code) == ewm_squared_values.end():
            old_squared_value = nan
        else:
            old_squared_value = ewm_squared_values[code]
        if ewm_cubed_values.find(code) == ewm_cubed_values.end():
            old_cubed_value = nan
        else:
            old_cubed_value = ewm_cubed_values[code]
        if ewm_fourth_values.find(code) == ewm_fourth_values.end():
            old_fourth_value = nan
        else:
            old_fourth_value = ewm_fourth_values[code]

        if np.isnan(old_value):
            # First occurrence: initialize EWM with current values.
            new_value = value
            new_squared_value = squared_value
            new_cubed_value = cubed_value
            new_fourth_value = fourth_value
            kurt = 0.0
        else:
            new_value = old_value * (1.0 - alpha) + value * alpha
            new_squared_value = old_squared_value * (1.0 - alpha) + squared_value * alpha
            new_cubed_value = old_cubed_value * (1.0 - alpha) + cubed_value * alpha
            new_fourth_value = old_fourth_value * (1.0 - alpha) + fourth_value * alpha

            variance = new_squared_value - new_value * new_value
            if variance > 1e-14:
                m4 = new_fourth_value - 4.0 * new_cubed_value * new_value + 6.0 * new_squared_value * new_value * new_value - 3.0 * new_value**4
                kurt = m4 / (variance * variance) - 3.0
            else:
                kurt = 0.0

        # Save updated state for this code
        ewm_values[code] = new_value
        ewm_squared_values[code] = new_squared_value
        ewm_cubed_values[code] = new_cubed_value
        ewm_fourth_values[code] = new_fourth_value
        last_timestamps[code] = ts

        result[i] = kurt

    return result

#########################################################
# Cython version of EwmKurtStepper using unordered_map
#########################################################


cdef class EwmKurtStepper:
    cdef dict __dict__
    
    cdef public int window
    cdef public double alpha
    cdef unordered_map[int64_t, double] ewm_values
    cdef unordered_map[int64_t, double] ewm_squared_values
    cdef unordered_map[int64_t, double] ewm_cubed_values
    cdef unordered_map[int64_t, double] ewm_fourth_values
    cdef unordered_map[int64_t, int64_t] last_timestamps

    def __init__(self, folder="", name="", window=1):
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.window = window
        self.alpha = get_alpha(window)
        self.ewm_values = unordered_map[int64_t, double]()
        self.ewm_squared_values = unordered_map[int64_t, double]()
        self.ewm_cubed_values = unordered_map[int64_t, double]()
        self.ewm_fourth_values = unordered_map[int64_t, double]()
        self.last_timestamps = unordered_map[int64_t, int64_t]()

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, int window=1):
        return cls.load_utility(cls, folder=folder, name=name, window=window)

    def __getstate__(self):
        """
        For pickling, convert each unordered_map to a Python dict.
        """
        state = self.__dict__.copy()
        state["ewm_values"] = _umap_to_pydict(self.ewm_values)
        state["ewm_squared_values"] = _umap_to_pydict(self.ewm_squared_values)
        state["ewm_cubed_values"] = _umap_to_pydict(self.ewm_cubed_values)
        state["ewm_fourth_values"] = _umap_to_pydict(self.ewm_fourth_values)
        state["last_timestamps"] = _umap_to_pydict_int(self.last_timestamps)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ewm_values = _pydict_to_umap(state["ewm_values"])
        self.ewm_squared_values = _pydict_to_umap(state["ewm_squared_values"])
        self.ewm_cubed_values = _pydict_to_umap(state["ewm_cubed_values"])
        self.ewm_fourth_values = _pydict_to_umap(state["ewm_fourth_values"])
        self.last_timestamps = _pydict_to_umap_int(state["last_timestamps"])

    def update(self, dt, dscode, serie):
        """
        Update EWM kurtosis values.

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        """
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        # Convert dscode and serie to memoryviews
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        return update_ewmkurt_values(codes, values, ts_array, self.alpha,
                                     self.ewm_values,
                                     self.ewm_squared_values,
                                     self.ewm_cubed_values,
                                     self.ewm_fourth_values,
                                     self.last_timestamps)

#############################################
# Helper functions for pickling support
# https://stackoverflow.com/questions/51686143/cython-iterate-through-map
#############################################
cdef dict _umap_to_pydict(unordered_map[int64_t, double] umap):
    cdef dict d = {}
    cdef unordered_map[int64_t, double].iterator it = umap.begin()
    cdef unordered_map[int64_t, double].iterator end_it = umap.end()
    while it != end_it:
        d[dereference(it).first] = dereference(it).second
        postincrement(it)
    return d

cdef dict _umap_to_pydict_int(unordered_map[int64_t, int64_t] umap):
    cdef dict d = {}
    cdef unordered_map[int64_t, int64_t].iterator it = umap.begin()
    cdef unordered_map[int64_t, int64_t].iterator end_it = umap.end()
    while it != end_it:
        d[dereference(it).first] = dereference(it).second
        postincrement(it)
    return d

cdef unordered_map[int64_t, double] _pydict_to_umap(dict d):
    cdef unordered_map[int64_t, double] umap = unordered_map[int64_t, double]()
    cdef int64_t key
    cdef double value
    for key, value in d.items():
        umap.insert((key, value))
    return umap

cdef unordered_map[int64_t, int64_t] _pydict_to_umap_int(dict d):
    cdef unordered_map[int64_t, int64_t] umap = unordered_map[int64_t, int64_t]()
    cdef int64_t key, value
    for key, value in d.items():
        umap.insert((key, value))
    return umap
