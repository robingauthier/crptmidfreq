
import numpy as np
import pandas as pd
import os
import pickle
from tdigest import TDigest
from crptmidfreq.stepper.base_stepper import BaseStepper

###############################################################################
# 1) A "QuantileStepper" class that updates a T-Digest for each code (symbol).
#    We then retrieve current quantile estimates as new data arrives.
#  Under the hood it performs a Kmean clustering
###############################################################################

class QuantileStepper(BaseStepper):
    """
    Maintains streaming quantile estimates (expanding window) for multiple codes
    via T-Digest. Each code has its own T-Digest. On each update, we merge the
    new batch of values into the respective T-Digest, then query multiple
    quantiles for each row.
    """
    def __init__(self, folder='', name='', qs=None):
        """
        Parameters
        ----------
        folder : str
            Where to save state (Pickle).
        name : str
            Name for the instance (file naming).
        qs : list of float
            List of quantiles, e.g. [0.1, 0.5, 0.9].
        """
        super().__init__(folder,name)
        if qs is None:
            qs = [0.5]  # default: median only
        self.qs = qs
        
        # code -> TDigest object
        self.tdigest_map = {}
        
    def update(self, dt, dscode, serie):
        """
        Update the T-Digest state with new data in (datetime, code, value) arrays.
        Return a 2D array (n_rows x len(qs)) of quantile values for each row.
        """
        n = len(serie)
        n_qs = len(self.qs)
        results = np.zeros((n, n_qs), dtype=float)
        
        for i in range(n):
            code = dscode[i]
            val = serie[i]
            if code not in self.tdigest_map:
                self.tdigest_map[code] = TDigest()
            # Update T-Digest
            self.tdigest_map[code].update(val)
            # Retrieve current quantiles
            for j, qv in enumerate(self.qs):
                results[i, j] = self.tdigest_map[code].percentile(qv*100)
        
        return results
    
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, qs=None):
        """Load instance from saved state or create new if not exists"""
        return QuantileStepper.load_utility(cls,folder=folder,name=name,qs=qs)

