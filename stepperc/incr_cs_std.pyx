# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os
import numpy as np
cimport numpy as np
from libc.math cimport isnan, sqrt, pow
from libcpp.unordered_map cimport unordered_map
from crptmidfreq.config_loc import get_feature_folder
from cython.operator import dereference, postincrement
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

#########################################################
# Define C++ structs to hold all state for one code
#########################################################
cdef struct StdTimestampState:
    int64_t timestamp
    
cdef struct StdByState:
    double sum
    double sum2
    double weight
    int64_t count

#########################################################
# Optimized update function using combined state_maps
#########################################################
cdef update_cs_std_values(int64_t[:] codes,
                          double[:] values,
                          int64_t[:] bys,
                          double[:] wgts,
                          int64_t[:] timestamps,
                          unordered_map[int64_t, StdTimestampState]& ts_state_map,
                          unordered_map[int64_t, StdByState]& by_state_map,
                          int mincnt):
    """
    Cross-sectional standard deviation calculation
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, j, code, by, ts, g_last_ts = 0
    cdef double value, wgt, vart
    cdef StdTimestampState* ts_state
    cdef StdByState* by_state
    cdef unordered_map[int64_t, StdByState].iterator it2

    # Get global last timestamp
    cdef unordered_map[int64_t, StdTimestampState].iterator it = ts_state_map.begin()
    while it != ts_state_map.end():
        g_last_ts = max(g_last_ts, (<int64_t>(dereference(it).second).timestamp))
        postincrement(it)
    
    j = 0
    for i in range(n):
        code = codes[i]
        value = values[i]
        by = bys[i]
        wgt = wgts[i]
        ts = timestamps[i]
        
        # Initialize by state if needed
        by_state = &by_state_map[by]
        
        # Check timestamp is increasing for this code
        ts_state = &ts_state_map[code]
        if ts_state.timestamp != 0 and ts < ts_state.timestamp:
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(ts_state.timestamp)
            raise ValueError("DateTime must be strictly increasing per code")
            
        # Check timestamp is increasing across instruments
        if ts < g_last_ts:
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('g_last_ts')
            print(g_last_ts)
            raise ValueError("DateTime must be strictly increasing across instruments")
            
        if ts != g_last_ts:
            # Process stored results up to this point
            while j < i:
                by2 = bys[j]
                if by_state_map[by2].weight > 0 and by_state_map[by2].count >= mincnt:
                    # Calculate variance with Bessel's correction
                    vart = (by_state_map[by2].sum2 / by_state_map[by2].weight - 
                           pow(by_state_map[by2].sum / by_state_map[by2].weight, 2))
                    vart = vart * (by_state_map[by2].count / (by_state_map[by2].count - 1))
                    result[j] = sqrt(vart) if vart > 0 else 0.0
                else:
                    result[j] = 0.0
                j += 1
                
            # Reset all by states
            it2 = by_state_map.begin()
            while it2 != by_state_map.end():
                (<StdByState>(dereference(it2).second)).sum = 0.0
                (<StdByState>(dereference(it2).second)).sum2 = 0.0
                (<StdByState>(dereference(it2).second)).weight = 0.0
                (<StdByState>(dereference(it2).second)).count = 0
                postincrement(it2)
                
        # Store updates
        ts_state.timestamp = ts
        g_last_ts = ts
        by_state.sum += value * wgt
        by_state.sum2 += value * value * wgt
        by_state.weight += wgt
        by_state.count += 1
        
    # Process remaining items
    while j < n:
        by2 = bys[j]
        if by_state_map[by2].weight > 0 and by_state_map[by2].count >= mincnt:
            # Calculate variance with Bessel's correction
            vart = (by_state_map[by2].sum2 / by_state_map[by2].weight - 
                   pow(by_state_map[by2].sum / by_state_map[by2].weight, 2))
            vart = vart * (by_state_map[by2].count / (by_state_map[by2].count - 1))
            result[j] = sqrt(vart) if vart > 0 else 0.0
        else:
            result[j] = 0.0
        j += 1
        
    return result

#########################################################
# Helper functions for pickling the state_maps
#########################################################
# Convert state_maps to Python dicts
cdef dict _umap_to_pydict_ts_state(unordered_map[int64_t, StdTimestampState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, StdTimestampState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = (<int64_t>(dereference(it).second).timestamp)
        postincrement(it)
    return d

cdef dict _umap_to_pydict_by_state(unordered_map[int64_t, StdByState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, StdByState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = (
            (<double>(dereference(it).second).sum),
            (<double>(dereference(it).second).sum2),
            (<double>(dereference(it).second).weight),
            (<int64_t>(dereference(it).second).count)
        )
        postincrement(it)
    return d

cdef unordered_map[int64_t, StdTimestampState] _pydict_to_umap_ts_state(dict d):
    cdef unordered_map[int64_t, StdTimestampState] state_map
    cdef int64_t key, ts
    cdef StdTimestampState s
    for key, ts in d.items():
        s.timestamp = ts
        state_map[key] = s
    return state_map

cdef unordered_map[int64_t, StdByState] _pydict_to_umap_by_state(dict d):
    cdef unordered_map[int64_t, StdByState] state_map
    cdef int64_t key
    cdef tuple t
    cdef StdByState s
    for key, t in d.items():
        s.sum = t[0]
        s.sum2 = t[1]
        s.weight = t[2]
        s.count = t[3]
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython csStdStepper
#########################################################
cdef class csStdStepper:
    cdef dict __dict__
    
    cdef public int mincnt
    cdef unordered_map[int64_t, StdTimestampState] ts_state_map
    cdef unordered_map[int64_t, StdByState] by_state_map

    def __init__(self, folder="", name="", mincnt=1):
        """
        Initialize csStdStepper for cross-sectional standard deviation
        """
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.mincnt = mincnt
        self.ts_state_map = unordered_map[int64_t, StdTimestampState]()
        self.by_state_map = unordered_map[int64_t, StdByState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, mincnt=1):
        """
        Load an instance of the class from a pickle file.
        """
        return load_instance(cls, folder, name, mincnt=mincnt)

    def __getstate__(self):
        """
        For pickling, convert state_maps to Python dicts.
        """
        state = self.__dict__.copy()
        state["ts_state_map"] = _umap_to_pydict_ts_state(self.ts_state_map)
        state["by_state_map"] = _umap_to_pydict_by_state(self.by_state_map)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ts_state_map = _pydict_to_umap_ts_state(state["ts_state_map"])
        self.by_state_map = _pydict_to_umap_by_state(state["by_state_map"])

    def update(self, dt, dscode, serie, by=None, wgt=None):
        """
        Update cross-sectional standard deviation values

        dt: numpy array of datetime64 values.
        dscode: numpy array of int64 codes.
        serie: numpy array of float64 values.
        by: numpy array of by values (all, sector, cluster). Defaults to array of ones.
        wgt: numpy array of weights [0-1]. Defaults to array of ones.
        """
        if wgt is None:
            wgt = np.ones_like(serie)
        if by is None:
            by = np.ones_like(serie)
            
        # Ensure minimum count for standard deviation calculation 
        self.mincnt = max(2, self.mincnt)
            
        # Ensure by is int64
        by_int = by.astype(np.int64)
        
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        cdef int64_t[:] by_values = by_int
        cdef double[:] weights = wgt
        
        res = update_cs_std_values(codes, values, by_values, weights, ts_array, 
                                  self.ts_state_map, self.by_state_map, self.mincnt)
        return np.array(res)