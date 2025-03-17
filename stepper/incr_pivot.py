import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from .base_stepper import BaseStepper


@njit
def incremental_pivot(timestamps, codes, values,last_timestamps,result):
    """
    Incremental pivot function that organizes data by unique timestamps and 
    assigns each dscode as a separate column without using pandas.
    """

    ucodes = np.unique(codes)
    udts = np.sort(np.unique(timestamps))
    ndts = len(udts)
    
    # initialisation
    for code in ucodes:
        result[np.int64(code)] = np.zeros(ndts)
        for k in range(ndts):
            result[np.int64(code)][k]=np.nan

    g_last_timestamp=0
    g_last_cnt=-1 # start at -1
    for i in range(len(codes)):
        code = np.int64(codes[i])
        value = values[i]
        ts = timestamps[i]
        
        
        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  
            raise ValueError("DateTime must be strictly increasing per code")
        if ts < g_last_timestamp:  
            raise ValueError("DateTime must be strictly increasing globally")
        
        if ts>g_last_timestamp:
            g_last_cnt+=1
            g_last_timestamp=ts
            
        result[code][g_last_cnt]=value

    return udts,result



class PivotStepper(BaseStepper):
    """Like .pivot_table in pandas"""

    def __init__(self, folder='', name=''):
        """
        """
        super().__init__(folder,name)

        # Initialize empty state
        self.last_cnts = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return PivotStepper.load_utility(cls,folder=folder,name=name)


    def update(self, dt, dscode, serie):
        """

        """
        self.validate_input(dt,dscode,serie)
        # Update values and timestamps using numba function
        res = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        print('pivot_start')
        udts,res= incremental_pivot(
            dt.view(np.int64),dscode.view(np.int64), serie, 
            self.last_timestamps,res)
        #rd = dict(res) # numba dict-> python dict
        #rd['dt']=udts
        print('pivot_end')
        return udts,res
