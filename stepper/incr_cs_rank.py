import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
import pickle
from .base_stepper import BaseStepper


@njit
def update_cs_rank_values(codes, values, bys,wgts,timestamps,
                         last_timestamps,last_vals,last_ranks,last_cnts):
    """
    """
    result = np.empty(len(codes), dtype=np.float64)

    g_last_ts = 0
    for k,v in last_timestamps.items():
        g_last_ts = max(v,g_last_ts)
        
    # j is a second iterator
    j=0 
    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        by = bys[i]
        wgt = wgts[i]
        ts = timestamps[i]
        
        if by not in last_vals:
            last_vals[by]=np.empty(0,dtype=np.float64)
            last_ranks[by]=np.empty(0,dtype=np.int64)
            last_cnts[by]=0

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Important to have strictly increasing 
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing per code")

        # Check timestamp is increasing accross dscode
        if ts < g_last_ts:  # Important to have strictly increasing 
            print('code')
            print(code)
            print('ts')
            print(ts)
            print('last_ts')
            print(last_ts)
            raise ValueError("DateTime must be strictly increasing accross instruments")

        if ts!=g_last_ts:
            for k,v in last_vals.items():
                last_ranks[k] = np.argsort(last_vals[k]).argsort()
            while j<i:
                # Store result for this row
                by2 = bys[j] 
                if last_cnts[by2]<=1:
                    result[j]=0.0
                else:
                    result[j] =(last_ranks[by2][0] - (last_cnts[by2]-1)/2)/(last_cnts[by2]-1)*2
                #result[j] =last_ranks[by2][0]
                if len(last_ranks[by2])>0:
                    last_ranks[by2] = last_ranks[by2][1:]
                j+=1
            # reset everything
            for by,v in last_vals.items():
                last_vals[by]=np.empty(0,dtype=np.float64)
                last_cnts[by]=0

        # Store updates
        last_timestamps[code] = ts
        g_last_ts= ts
        last_vals[by]=np.concatenate((last_vals[by], np.array([value * wgt],dtype=np.float64)))
        last_cnts[by]+=1
        
    # the last value is not assigned 
    by2 = bys[j] 
    return result



class csRankStepper(BaseStepper):


    def __init__(self, folder='', name=''):
        super().__init__(folder,name)

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
    def load(cls, folder, name, window=1):
        """Load instance from saved state or create new if not exists"""
        return csRankStepper.load_utility(cls,folder=folder,name=name)

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
                         self.last_timestamps,self.last_vals,self.last_ranks,self.last_cnts)

