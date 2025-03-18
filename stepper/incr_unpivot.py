import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper


#@njit
def incremental_unpivot(timestamps, pserie):
    """
    Incremental pivot function that organizes data by unique timestamps and 
    assigns each dscode as a separate column without using pandas.
    """
    
    ncodes = len(pserie)
    ndts = timestamps.shape[0]
    n = ncodes*ndts
    
    ndt = np.empty(n,dtype=np.int64)
    ndscode = np.empty(n,dtype=np.int64)
    nserie = np.empty(n,dtype=np.float64)
    
    icode=-1
    for code,serie in pserie.items():
        icode+=1
        for j in range(ndts):
            k = j*ncodes+icode
            print(k)
            ndt[k]=timestamps[j]
            ndscode[k]=code
            nserie[k]=serie[j]
    return ndt,ndscode,nserie



class UnPivotStepper(BaseStepper):

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
        return UnPivotStepper.load_utility(cls,folder=folder,name=name)


    def update(self, dt, pserie):
        """

        """
        assert isinstance(pserie,Dict),'not a numba dict'
        for k,v in pserie.items():
            assert len(dt)==len(v),f'issue len of {k}'

        ndt,ndscode,nserie= incremental_unpivot(
            dt.view(np.int64),pserie)
        return ndt,ndscode,nserie
