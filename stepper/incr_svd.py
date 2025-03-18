import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper


#@njit
def incremental_svd(pseries,puniv,last_mem,last_univ,mempos,warmpos,fitfreq,n_comp,n_comp_var):
    """
    Incremental pivot function that organizes data by unique timestamps and 
    assigns each dscode as a separate column without using pandas.
    
    X = U D V.T
    """
    
    n = pseries.shape[0]
    lookback = last_mem.shape[0]
    result_resid = np.zeros(pseries.shape,dtype=np.float64) # residuals of pseries
    result_D = np.zeros(pseries.shape,dtype=np.float64) # how much each factor explains risk
    result_FR  = np.zeros(pseries.shape,dtype=np.float64) # factor returns

    last_Vt = None
    last_Dinv = None

    for i in range(n):
        mempos=(mempos+1)%lookback# position in last_mem . It is circular
        warmpos+=1 # min nb of points to compute svd

        last_mem[mempos,:]=pseries[i,:]
        change_univ = np.all(puniv[i,:]==last_univ)
        last_univ = puniv[i,:]


        if warmpos<lookback:
            continue
        
        recompute_cond = (last_Vt is None or i%fitfreq==0 or change_univ)
        
        if recompute_cond :
            U, D, Vt = np.linalg.svd(last_mem[:,last_univ>0], full_matrices=False)

            is_sorted_desc = np.all(np.diff(D) <= 0)
            assert is_sorted_desc,f'issue SVD sorting for i={i}'
            last_U = U
            last_D = D
            last_Vt = Vt
            
            last_Dinv= np.where(last_D>0,1/last_D,0)
            last_V = np.transpose(Vt)
        else:
            # X = U D Vt   D and Vt  will not change
            # U = X D-1 V
            last_U = last_mem[:,last_univ>0] @ np.diag(last_Dinv) @ last_V
            last_D = D
            last_Vt = Vt
            
            
        if n_comp>0:
            n_comp_loc = n_comp
        else:
            n_comp_loc = n_comp_var # TODO: use D to get pct of variance
        
        # computing the truncated X
        n_min = min(last_D.shape[0],np.sum(last_univ))
        last_UD = last_U @ np.diag(last_D)
        U_trunc = last_U[:,:n_comp_loc]
        D_trunc = last_D[:n_comp_loc]
        Vt_trunc = last_Vt[:n_comp_loc,:]
        
        # computing an approximation
        X_ap = U_trunc @ np.diag(D_trunc)@Vt_trunc
        X_resid=last_mem[:,last_univ>0] - X_ap
        
        # Setting the results
        result_resid[i,last_univ>0]=X_resid[mempos]
        result_D[i,:n_min] = last_D[:n_min]
        result_FR[i,:n_min] = last_UD[mempos,:n_min]
    
    # residuals, % variance explained, Factor returns
    return result_resid,result_D,result_FR



class SVDStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='',lookback=300,fitfreq=10,n_comp=2,n_comp_var=0):
        """
        """
        super().__init__(folder,name)
        self.n_comp=n_comp
        self.n_comp_var=n_comp_var
        self.fitfreq=fitfreq
        self.lookback=lookback
        # Initialize empty state
        self.last_mem = None
        self.mempos = 0
        self.warmpos=0
        self.last_mem=None
        self.last_univ=None
        
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return SVDStepper.load_utility(cls,folder=folder,name=name)


    def update(self, dt, pserie, puniv=None):
        """
        pserie : pivoted serie
        puniv : pivoted PIT universe with 1/0
        """
        if not isinstance(pserie,np.ndarray):
            # create a 2 dim array from pserie that
            pserie=np.stack([v for k,v in pserie.items()]).T

        if puniv is None:
            puniv=np.ones_like(pserie)        
        
        if self.last_mem is None:
            self.last_mem=np.zeros((self.lookback,pserie.shape[1]),dtype=np.float64)
        if self.last_univ is None:
            self.last_univ=np.zeros(pserie.shape[1],dtype=np.int64)
        
        # pseries,puniv,last_mem,mempos,warmpos,fitfreq,n_comp,n_comp_var
        result_resid,result_D,result_FR= incremental_svd(
            pserie,puniv,
            self.last_mem,
            self.last_univ,
            self.mempos,
            self.warmpos,
            self.fitfreq,
            self.n_comp,
            self.n_comp_var)
        return result_resid,result_D,result_FR
