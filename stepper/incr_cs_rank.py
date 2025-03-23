import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
import pickle
from crptmidfreq.stepper.base_stepper import BaseStepper


@njit
def update_cs_rank_values(codes, values, bys, wgts, timestamps,
                          last_timestamps, last_vals, last_ranks, last_cnts, percent):
    result = np.empty(len(codes), dtype=np.float64)
    
    g_last_ts = 0
    # Set block_start to mark the beginning of the current block of rows.
    block_start = 0
    
    # First, get the global last timestamp from last_timestamps.
    for k, v in last_timestamps.items():
        if v > g_last_ts:
            g_last_ts = v

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        by = bys[i]
        wgt = wgts[i]
        ts = timestamps[i]

        if by not in last_vals:
            last_vals[by] = np.empty(0, dtype=np.float64)
            last_ranks[by] = np.empty(0, dtype=np.int64)
            last_cnts[by] = 0

        # Check timestamp is strictly increasing for each code.
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:
            raise ValueError("DateTime must be strictly increasing per code")
        if ts < g_last_ts:
            raise ValueError("DateTime must be strictly increasing across instruments")
        
        # If we detect a new timestamp compared to the global one,
        # process the current block.
        if ts != g_last_ts:
            # Update last_ranks for each 'by'
            for k, v in last_vals.items():
                last_ranks[k] = np.argsort(last_vals[k]).argsort()
            for j in range(block_start, i):
                by2 = bys[j]
                if last_cnts[by2] <= 1:
                    result[j] = 0.0
                else:
                    if percent==0:
                        result[j] = (last_ranks[by2][0] - (last_cnts[by2]-1)/2) / (last_cnts[by2]-1)*2
                    elif percent==1:
                        result[j] = last_ranks[by2][0]
                    else:
                        result[j] = last_cnts[by2]-last_ranks[by2][0]
                # Optionally, drop the first element of last_ranks[by2]
                if len(last_ranks[by2]) > 0:
                    last_ranks[by2] = last_ranks[by2][1:]
            # Reset for new block.
            for k, v in last_vals.items():
                last_vals[k] = np.empty(0, dtype=np.float64)
                last_cnts[k] = 0
            block_start = i  # new block starts at current i

        # Store updates for the current row.
        last_timestamps[code] = ts
        g_last_ts = ts
        # Append new value.
        last_vals[by] = np.concatenate((last_vals[by], np.array([value * wgt], dtype=np.float64)))
        last_cnts[by] += 1

    # Process any remaining rows at the end.
    if block_start < len(codes):
        for k, v in last_vals.items():
            last_ranks[k] = np.argsort(last_vals[k]).argsort()
        for j in range(block_start, len(codes)):
            by2 = bys[j]
            if last_cnts[by2] <= 1:
                result[j] = 0.0
            else:
                if percent==0:
                    result[j] = (last_ranks[by2][0] - (last_cnts[by2]-1)/2) / (last_cnts[by2]-1)*2
                elif percent==1:
                    result[j] = last_ranks[by2][0]
                else:
                    result[j] = last_cnts[by2]-last_ranks[by2][0]
            if len(last_ranks[by2]) > 0:
                last_ranks[by2] = last_ranks[by2][1:]
    return result


class csRankStepper(BaseStepper):


    def __init__(self, folder='', name='',percent=0):
        super().__init__(folder,name) 
        self.percent = percent 
        # Initialize empty state
        self.last_vals = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_ranks = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.int64, 1, 'C')
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.last_cnts = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name,percent=0):
        """Load instance from saved state or create new if not exists"""
        return csRankStepper.load_utility(cls,folder=folder,name=name,percent=percent)

    def update(self, dt, dscode, serie,by=None,wgt=None):
        """
        Update difference values for each code and return the difference values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process
            by : numpy array of by values ( all, sector, cluster)
            wgt : numpy array of weight/univ flag [0-1]

        Returns:
            numpy array of same length as input arrays containing difference values
        """
        if wgt is None:
            wgt = np.ones_like(serie)
        if by is None:
            by = np.ones_like(serie)

        self.validate_input(dt,dscode,serie,by=by,wgt=wgt)

        # by must be integers 
        by = by.astype('int64')
        
        # Update values and timestamps using numba function
        return update_cs_rank_values(dscode, serie, by,wgt,dt.view(np.int64),
                         self.last_timestamps,self.last_vals,self.last_ranks,self.last_cnts,
                         self.percent)

