import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from .base_stepper import BaseStepper

# Quite similar to the featurelib/timeclf.py



class ModelStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='',lookback=300,minlookback=100,
                 fitfreq=10,gap=1,model_gen=None,with_fit=True):
        """
        """
        super().__init__(folder,name)
        self.fitfreq=fitfreq
        self.gap=gap
        self.lookback=lookback
        self.minlookback=minlookback
        self.model_gen = model_gen# you need to call model_gen() to create a new model 
        self.with_fit=with_fit
        
        # Initialize empty state
        self.last_xmem = None
        self.last_ymem = None
        self.last_wmem = None
        self.mempos = 0
        self.warmpos=0
        self.last_i =0 # when the last model was fit
        self.last_mem=None
        self.last_univ=None
        
        # History of models
        self.hmodels=[]
        
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name,lookback=300,minlookback=100,
                 fitfreq=10,gap=1,model_gen=None,with_fit=True):
        """Load instance from saved state or create new if not exists"""
        return ModelStepper.load_utility(cls,folder=folder,name=name,
                                         lookback=lookback,minlookback=minlookback,
                 fitfreq=fitfreq,gap=gap,
                 model_gen=model_gen,with_fit=with_fit
                                         )


    def update(self, dts, xseries, yserie=None,wgtserie=None):
        """
        """
        assert isinstance(xseries,np.ndarray)
        if yserie is not None:
            assert xseries.shape[0]==yserie.shape[0]

        result = np.zeros(xseries.shape[0],dtype=np.float64)        
        if self.last_xmem is None:
            self.last_xmem=np.zeros((self.lookback,xseries.shape[1]),dtype=np.float64)
            self.last_ymem=np.zeros(self.lookback,dtype=np.float64)
            self.last_wmem=np.zeros(self.lookback,dtype=np.float64)
        model_loc=None
        n=xseries.shape[0]
        for i in range(n):
            self.mempos=(self.mempos+1)%self.lookback # position in memory . It is circular
            self.warmpos+=1 # min nb of points to compute svd
            self.last_i+=1

            if self.with_fit:
                self.last_xmem[self.mempos,:]=xseries[i,:]
                self.last_ymem[self.mempos]=yserie[i]
                self.last_wmem[self.mempos]=wgtserie[i]
                
                if (self.warmpos>=self.minlookback) and (i%self.fitfreq==0):
                    model_loc = self.model_gen()
                    model_loc.fit(X=self.last_xmem,y=self.last_ymem,sample_weight=self.last_wmem)
                    self.last_i=0
                    self.hmodels+=[{'dt':dts[i],'i':self.last_i,'model':model_loc}]
            else:
                model_loc=self.hmodels[-1]
            
            if model_loc is not None:
                # caution we are not using the gap here !!!
                ypred=model_loc.predict(xseries[[i],:])
            else:
                ypred = np.nan
            
            result[i]=ypred
        return result
