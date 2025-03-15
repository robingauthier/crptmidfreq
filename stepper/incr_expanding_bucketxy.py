import numpy as np
import pandas as pd
import os
import pickle
from tdigest import TDigest
from .base_stepper import BaseStepper

# ipython -i stepper/incr_expanding_bucketxy.py


class BucketXYStepper(BaseStepper):
    """
    Maintains streaming quantile estimates for x using T-Digest, then produces a
    bucket plot of y vs. x by quantile-based buckets.
    """
    def __init__(self, folder='', name='', n_buckets=8):
        """
        Parameters
        ----------
        folder : str
            Where to save state (Pickle).
        name : str
            Name for the instance (file naming).
        n_buckets : int
            Number of buckets (e.g. 8 => boundaries at [1/8, 2/8, ..., 7/8]).
        """
        super().__init__(folder, name)
        self.n_buckets = n_buckets

        # Single T-Digest for x (if you'd like code-by-code, store a map here).
        self.tdigest_map = TDigest()
        
        self.qtls_x = {k:np.nan for k in range(1,self.n_buckets)}
        self.bucket_cnt = {k:0 for k in range(self.n_buckets)}
        self.bucket_sum = {k:0 for k in range(self.n_buckets)}
        self.bucket_sum2 = {k:0 for k in range(self.n_buckets)}


    def update(self, dt, dscode, x_values, y_values):
        """
        Incorporate new points (x, y) into T-Digest, then compute a bucket plot
        for all data so far (expanding). Return a DataFrame of shape (n_buckets).
        
        * dt, dscode: not used here in the basic aggregator. 
                     Shown just to respect the 'Stepper' signature.
        
        Parameters
        ----------
        x_values : array-like
        y_values : array-like
        
        Returns
        -------
        df_buckets : pd.DataFrame with columns:
            'bucket_index', 'x_left', 'x_right',
            'mean_y', 'stderr_y', 'count'
        """
        self.validate_input(dt,dscode,x_values,y_values=y_values)
        
        n = dscode.shape[0]
        results_mean = np.zeros((n, self.n_buckets), dtype=np.float64)
        results_std = np.zeros((n, self.n_buckets), dtype=np.float64) # std / np.sqrt(n) 
        
        for i in range(n):
            code = dscode[i]
            xval = x_values[i]
            yval = y_values[i]
            
            
            # Update T-Digest            
            self.tdigest_map.update(xval)

            # Retrieve current quantiles
            for j in range(self.n_buckets):
                self.qtls_x[j] = self.tdigest_map.percentile((j+1)*100/(self.n_buckets+1))
        
            # Update bucket stats
            bucket_index = None
            for j in range(self.n_buckets):
                if j==0:
                    left = -np.inf
                else:
                    left = self.qtls_x[j-1]
                if j==self.n_buckets-1:
                    right = np.inf
                else:
                    right = self.qtls_x[j]
                if np.isnan(left) or np.isnan(right):
                    break
                if left <= xval < right:
                    bucket_index = j
                    break
            
            # Accumulate stats
            if bucket_index is not None:
                self.bucket_cnt[bucket_index]+=1
                self.bucket_sum[bucket_index]+=yval
                self.bucket_sum2[bucket_index]+=yval*yval

            # 6) Compute mean and standard error in each bucket
            for j in range(self.n_buckets):
                if self.bucket_cnt[j]==0:
                    results_mean[i][j]=np.nan
                    results_std[i][j] =np.nan
                    continue
                e_x2 = self.bucket_sum2[j]/self.bucket_cnt[j]
                e_x = self.bucket_sum[j]/self.bucket_cnt[j]
                results_mean[i][j] = e_x
                results_std[i][j]  = (e_x2 - e_x*e_x)/np.sqrt(self.bucket_cnt[j])
        return results_mean,results_std

    def save(self):
        """Pickle the entire object to disk."""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        path = os.path.join(self.folder, f"{self.name}_quantile.pkl")
        with open(path, 'wb') as f:
            pickle.dump({
                'n_buckets': self.n_buckets,
                'tdigest_map': self.tdigest_map,
                'qtls_x': self.qtls_x,
                'bucket_cnt': self.bucket_cnt,
                'bucket_sum': self.bucket_sum,
                'bucket_sum2': self.bucket_sum2,
            }, f)
    
    @classmethod
    def load(cls, folder, name):
        """Load from disk into a new QuantileStepper instance."""
        path = os.path.join(folder, f"{name}_quantile.pkl")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(folder=folder, name=name, n_buckets=data['n_buckets'])
        
        # restore T-Digest
        instance.tdigest_map = data['tdigest_map']
        instance.qtls_x = data['qtls_x']
        instance.bucket_cnt = data['bucket_cnt']
        instance.bucket_sum = data['bucket_sum']
        instance.bucket_sum2 = data['bucket_sum2']
        return instance
