
import numpy as np

from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.stepper.p2_algo.exp_qtl2 import expanding_quantile


class QuantileStepper(BaseStepper):
    """
    Maintains streaming quantile estimates (expanding window) for multiple codes
    via T-Digest. Each code has its own T-Digest. On each update, we merge the
    new batch of values into the respective T-Digest, then query multiple
    quantiles for each row.
    """
    def __init__(self, folder='', name='', qs=None,freq=int(60*24*5)):
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
        
        # Initialize empty state
        self.last_values = {}
        self.last_i = {}
        self.freq=freq # frenquency of update of the quantiles
        
    def update(self, dt, dscode, serie):
        """
        Update the T-Digest state with new data in (datetime, code, value) arrays.
        Return a 2D array (n_rows x len(qs)) of quantile values for each row.
        """
        n=len(dt)
        n_qs=len(self.qs)
        res  = np.zeros((n, n_qs), dtype=np.float64)
        expanding_quantile(dt.view(np.int64), 
                                 dscode.view(np.int64), 
                                 serie.view(np.float64), np.array(self.qs),
                                 self.tdigest_map,
                                 self.freq,
                                 self.last_values,self.last_i,
                                 res)
        return res
    
    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, qs=None):
        """Load instance from saved state or create new if not exists"""
        return QuantileStepper.load_utility(cls,folder=folder,name=name,qs=qs)

