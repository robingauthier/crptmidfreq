# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, language_level=3
import os

import numpy as np

cimport numpy as np
from libc.math cimport isnan
from libcpp.algorithm cimport sort
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from cython.operator import dereference, postincrement

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.stepperc.utils import load_instance, save_instance

# Typedef C types for clarity
ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

# Define a struct to track value and original index for sorting
cdef struct ValIndexPair:
    double value
    int index

# Custom comparison function for sorting ValIndexPair by value
cdef int compare_pairs(const ValIndexPair& a, const ValIndexPair& b) nogil:
    if a.value < b.value:
        return 1
    else:
        return 0

#########################################################
# Define C++ structs to hold all state for one code
#########################################################
cdef struct RankTimestampState:
    int64_t timestamp

cdef struct RankByState:
    vector[double] values
    vector[int64_t] ranks
    int64_t count

#########################################################
# Optimized update function using combined state_maps
#########################################################
cdef update_cs_rank_values(int64_t[:] codes,
                          double[:] values,
                          int64_t[:] bys,
                          double[:] wgts,
                          int64_t[:] timestamps,
                          unordered_map[int64_t, RankTimestampState]& ts_state_map,
                          unordered_map[int64_t, RankByState]& by_state_map,
                          int percent):
    """
    Cross-sectional rank calculation
    """
    cdef int64_t n = codes.shape[0]
    cdef double[:] result = np.empty(n, dtype=np.float64)
    cdef int64_t i, j, block_start = 0, code, by, by2, ts, g_last_ts = 0
    cdef double value, wgt
    cdef RankTimestampState* ts_state
    cdef RankByState* by_state
    
    # Helper vectors for sorting and ranking
    cdef vector[ValIndexPair] val_index_pairs
    cdef vector[int] ranks

    # Get global last timestamp
    cdef unordered_map[int64_t, RankTimestampState].iterator it = ts_state_map.begin()
    while it != ts_state_map.end():
        g_last_ts = max(g_last_ts, (<int64_t>(dereference(it).second).timestamp))
        postincrement(it)
    
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
            raise ValueError("DateTime must be strictly increasing per code")
            
        # Check timestamp is increasing across instruments
        if ts < g_last_ts:
            raise ValueError("DateTime must be strictly increasing across instruments")
            
        # If we detect a new timestamp, process the current block
        if ts != g_last_ts:
            # Update ranks for each 'by' group
            it_by = by_state_map.begin()
            while it_by != by_state_map.end():
                # Skip if no values in this group
                if (<RankByState>(dereference(it_by).second)).values.size() == 0:
                    postincrement(it_by)
                    continue
                    
                # Prepare for ranking: create vector of (value, index) pairs
                val_index_pairs.clear()
                val_index_pairs.resize((<RankByState>(dereference(it_by).second)).values.size())
                for j in range((<RankByState>(dereference(it_by).second)).values.size()):
                    val_index_pairs[j].value = (<RankByState>(dereference(it_by).second)).values[j]
                    val_index_pairs[j].index = j
                
                # Sort by value
                sort(val_index_pairs.begin(), val_index_pairs.end(), compare_pairs)
                
                # Assign ranks
                ranks.clear()
                ranks.resize(val_index_pairs.size())
                for j in range(val_index_pairs.size()):
                    ranks[val_index_pairs[j].index] = j
                
                # Store ranks
                (<RankByState>(dereference(it_by).second)).ranks.clear()
                for j in range(ranks.size()):
                    (<RankByState>(dereference(it_by).second)).ranks.push_back(ranks[j])
                    
                # Move to next element
                postincrement(it_by)
            
            # Calculate results for each row in the block
            for j in range(block_start, i):
                by2 = bys[j]
                
                if by_state_map[by2].count <= 1:
                    result[j] = 0.0
                else:
                    if by_state_map[by2].ranks.size() > 0:
                        if percent == 0:  # Normalized rank: -1 to 1
                            result[j] = (by_state_map[by2].ranks[0] - (by_state_map[by2].count - 1) / 2.0) / (by_state_map[by2].count - 1) * 2.0
                        elif percent == 1:  # Raw rank: 0 to N-1
                            result[j] = by_state_map[by2].ranks[0]
                        else:  # Reversed rank: N-1 to 0
                            result[j] = by_state_map[by2].count - by_state_map[by2].ranks[0] - 1
                            
                        # Remove first rank
                        if by_state_map[by2].ranks.size() > 0:
                            by_state_map[by2].ranks.erase(by_state_map[by2].ranks.begin())
                    else:
                        result[j] = 0.0
            
            # Reset for new block
            it_by = by_state_map.begin()
            while it_by != by_state_map.end():
                (<RankByState>(dereference(it_by).second)).values.clear()
                (<RankByState>(dereference(it_by).second)).count = 0
                postincrement(it_by)
            
            block_start = i  # new block starts at current i
                
        # Store updates
        ts_state.timestamp = ts
        g_last_ts = ts
        
        # Add value to appropriate by group
        by_state.values.push_back(value * wgt)
        by_state.count += 1
        
    # Process any remaining rows at the end
    if block_start < n:
        # Update ranks for each 'by' group
        it_by = by_state_map.begin()
        while it_by != by_state_map.end():
            # Skip if no values in this group
            if (<RankByState>(dereference(it_by).second)).values.size() == 0:
                postincrement(it_by)
                continue
                
            # Prepare for ranking: create vector of (value, index) pairs
            val_index_pairs.clear()
            val_index_pairs.resize((<RankByState>(dereference(it_by).second)).values.size())
            for j in range((<RankByState>(dereference(it_by).second)).values.size()):
                val_index_pairs[j].value = (<RankByState>(dereference(it_by).second)).values[j]
                val_index_pairs[j].index = j
            
            # Sort by value
            sort(val_index_pairs.begin(), val_index_pairs.end(), compare_pairs)
            
            # Assign ranks
            ranks.clear()
            ranks.resize(val_index_pairs.size())
            for j in range(val_index_pairs.size()):
                ranks[val_index_pairs[j].index] = j
            
            # Store ranks
            (<RankByState>(dereference(it_by).second)).ranks.clear()
            for j in range(ranks.size()):
                (<RankByState>(dereference(it_by).second)).ranks.push_back(ranks[j])
                
            # Move to next element
            postincrement(it_by)
        
        # Calculate results for each row in the block
        for j in range(block_start, n):
            by2 = bys[j]
            
            if by_state_map[by2].count <= 1:
                result[j] = 0.0
            else:
                if by_state_map[by2].ranks.size() > 0:
                    if percent == 0:  # Normalized rank: -1 to 1
                        result[j] = (by_state_map[by2].ranks[0] - (by_state_map[by2].count - 1) / 2.0) / (by_state_map[by2].count - 1) * 2.0
                    elif percent == 1:  # Raw rank: 0 to N-1
                        result[j] = by_state_map[by2].ranks[0]
                    else:  # Reversed rank: N-1 to 0
                        result[j] = by_state_map[by2].count - by_state_map[by2].ranks[0] - 1
                        
                    # Remove first rank
                    if by_state_map[by2].ranks.size() > 0:
                        by_state_map[by2].ranks.erase(by_state_map[by2].ranks.begin())
                else:
                    result[j] = 0.0
        
    return result

#########################################################
# Helper functions for pickling the state_maps
#########################################################
# Convert state_maps to Python dicts
cdef dict _umap_to_pydict_ts_state(unordered_map[int64_t, RankTimestampState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, RankTimestampState].iterator it = state_map.begin()
    while it != state_map.end():
        d[ (<int64_t>(dereference(it).first)) ] = (<int64_t>(dereference(it).second).timestamp)
        postincrement(it)
    return d

cdef dict _umap_to_pydict_by_state(unordered_map[int64_t, RankByState]& state_map):
    cdef dict d = {}
    cdef unordered_map[int64_t, RankByState].iterator it = state_map.begin()
    cdef vector[double] values_vec
    cdef vector[int64_t] ranks_vec
    cdef int i
    
    while it != state_map.end():
        values_vec = (<vector[double]>(dereference(it).second).values)
        ranks_vec = (<vector[int64_t]>(dereference(it).second).ranks)
        
        values_array = np.empty(values_vec.size(), dtype=np.float64)
        ranks_array = np.empty(ranks_vec.size(), dtype=np.int64)
        
        for i in range(values_vec.size()):
            values_array[i] = values_vec[i]
            
        for i in range(ranks_vec.size()):
            ranks_array[i] = ranks_vec[i]
            
        d[ (<int64_t>(dereference(it).first)) ] = (
            values_array,
            ranks_array,
            (<int64_t>(dereference(it).second).count)
        )
        postincrement(it)
    return d

cdef unordered_map[int64_t, RankTimestampState] _pydict_to_umap_ts_state(dict d):
    cdef unordered_map[int64_t, RankTimestampState] state_map
    cdef int64_t key, ts
    cdef RankTimestampState s
    for key, ts in d.items():
        s.timestamp = ts
        state_map[key] = s
    return state_map

cdef unordered_map[int64_t, RankByState] _pydict_to_umap_by_state(dict d):
    cdef unordered_map[int64_t, RankByState] state_map
    cdef int64_t key
    cdef tuple t
    cdef RankByState s
    cdef np.ndarray[double, ndim=1] values_array
    cdef np.ndarray[int64_t, ndim=1] ranks_array
    cdef int i
    
    for key, t in d.items():
        values_array = t[0]
        ranks_array = t[1]
        s.count = t[2]
        
        s.values.resize(values_array.shape[0])
        for i in range(values_array.shape[0]):
            s.values[i] = values_array[i]
            
        s.ranks.resize(ranks_array.shape[0])
        for i in range(ranks_array.shape[0]):
            s.ranks[i] = ranks_array[i]
            
        state_map[key] = s
    return state_map

#########################################################
# Optimized Cython csRankStepper
#########################################################
cdef class csRankStepper:
    cdef dict __dict__
    
    cdef public int percent
    cdef unordered_map[int64_t, RankTimestampState] ts_state_map
    cdef unordered_map[int64_t, RankByState] by_state_map

    def __init__(self, folder="", name="", percent=0):
        """
        Initialize csRankStepper for cross-sectional ranking
        """
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name
        self.percent = percent
        self.ts_state_map = unordered_map[int64_t, RankTimestampState]()
        self.by_state_map = unordered_map[int64_t, RankByState]()

    def save(self):
        save_instance(self)

    @classmethod
    def load(cls, folder, name, percent=0):
        """
        Load an instance of the class from a pickle file.
        """
        return load_instance(cls, folder, name, percent=percent)

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
        Update cross-sectional rank values

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
            
        # Validate inputs
        self.validate_input(dt, dscode, serie, by=by, wgt=wgt)
        
        # Ensure by is int64
        by_int = by.astype(np.int64)
        
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        cdef int64_t[:] codes = dscode
        cdef double[:] values = serie
        cdef int64_t[:] by_values = by_int
        cdef double[:] weights = wgt
        
        res = update_cs_rank_values(codes, values, by_values, weights, ts_array, 
                                   self.ts_state_map, self.by_state_map, self.percent)
        return np.array(res)