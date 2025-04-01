import numpy as np
import pandas as pd
import os
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.stepper.incr_expanding_quantile_tdigest import QuantileStepper
import matplotlib.pyplot as plt
from crptmidfreq.utils.common import get_logger
from crptmidfreq.config_loc import get_analysis_folder
logger = get_logger()


def try_to_save_png(tsave_graph_path):
    try:
        plt.savefig(tsave_graph_path)
        logger.info(f'Bktest Graph saved to {tsave_graph_path}')
    except Exception as e:
        try_to_save_png(tsave_graph_path.replace('.png', '_bis.png'))
    plt.close()


@njit
def compute_bucketxy(dt, dscode, x_values, y_values,
                     qtls_x, bucket_cnt, bucket_sum, bucket_sum2,
                     results_mean, results_std):
    n = dt.shape[0]
    n_buckets = qtls_x.shape[1]
    for i in range(n):
        code = dscode[i]
        xval = x_values[i]
        yval = y_values[i]

        if np.isnan(yval):
            continue
        if np.isnan(xval):
            continue

        # Update bucket stats
        bucket_index = None
        for j in range(n_buckets):
            if j == 0:
                left = -np.inf
            else:
                left = qtls_x[i][j-1]
            if j == n_buckets-1:
                right = np.inf
            else:
                right = qtls_x[i][j]
            if np.isnan(left) or np.isnan(right):
                break
            if left <= xval < right:
                bucket_index = j
                break

        # Accumulate stats
        if bucket_index is not None:
            bucket_cnt[bucket_index] += 1
            bucket_sum[bucket_index] += yval
            bucket_sum2[bucket_index] += yval*yval

        # 6) Compute mean and standard error in each bucket
        for j in range(n_buckets):
            if bucket_cnt[j] == 0:
                results_mean[i][j] = np.nan
                results_std[i][j] = np.nan
                continue
            e_x2 = bucket_sum2[j]/bucket_cnt[j]
            e_x = bucket_sum[j]/bucket_cnt[j]
            results_mean[i][j] = e_x
            results_std[i][j] = (e_x2 - e_x*e_x)/np.sqrt(bucket_cnt[j])


class BucketXYStepper(BaseStepper):
    """
    Maintains streaming quantile estimates for x using T-Digest, then produces a
    bucket plot of y vs. x by quantile-based buckets.
    """

    def __init__(self, folder='', name='', n_buckets=8, freq=int(60*24*5)):
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
        lqs = []
        for i in range(n_buckets):
            qsloc = (i+1)/(n_buckets+1)
            lqs += [qsloc]
        self.qutile_steppers = QuantileStepper()\
            .load(folder=folder,
                  name=name+'_qtl',
                  qs=lqs,
                  freq=freq)

        self.bucket_cnt = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.bucket_sum = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.bucket_sum2 = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )

        # Initialisation
        for k in range(self.n_buckets):
            self.bucket_cnt[k] = 0
            self.bucket_sum[k] = 0.0
            self.bucket_sum2[k] = 0.0

        with_plot = True
        if with_plot:
            self.save_graph_path = os.path.join(get_analysis_folder(),
                                                f'bucket_{self.name}.png')
        else:
            self.save_graph_path = None

    def save(self):
        self.qutile_steppers.save()
        self.save_utility(skip=['qutile_steppers'])

    @classmethod
    def load(cls, folder, name, n_buckets=8, freq=int(60*24*5)):
        """Load instance from saved state or create new if not exists"""
        r = BucketXYStepper.load_utility(cls, folder=folder, name=name,
                                         n_buckets=n_buckets, freq=freq)
        lqs = []
        for i in range(n_buckets):
            qsloc = (i+1)/(n_buckets+1)
            lqs += [qsloc]
        e = QuantileStepper.load(folder=folder, name=f'{name}_qtl', qs=lqs, freq=freq)
        r.qutile_steppers = e
        return r

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
        self.validate_input(dt, dscode, x_values, y_values=y_values)

        n = dscode.shape[0]
        results_mean = np.zeros((n, self.n_buckets), dtype=np.float64)
        results_std = np.zeros((n, self.n_buckets), dtype=np.float64)  # std / np.sqrt(n)

        # Retrieve current quantiles
        qtls_x = self.qutile_steppers.update(dt, dscode, x_values)
        compute_bucketxy(dt,
                         dscode, x_values, y_values,
                         qtls_x,
                         self.bucket_cnt,
                         self.bucket_sum,
                         self.bucket_sum2,
                         results_mean,
                         results_std)

        # --------------------------
        # Optional: Save Graphs and Daily PnL
        # --------------------------
        if self.save_graph_path is not None:
            # Plot cumulative net pnl (using daily_net from pandas aggregation)
            df_buckets = pd.DataFrame()
            df_buckets['x'] = range(self.n_buckets)
            df_buckets['cnt'] = pd.Series(self.bucket_cnt)
            df_buckets['sum'] = pd.Series(self.bucket_sum)
            df_buckets['sum2'] = pd.Series(self.bucket_sum2)
            df_buckets['avg'] = df_buckets['sum']/df_buckets['cnt']
            df_buckets['var'] = df_buckets['sum2']/df_buckets['cnt']-df_buckets['avg']*df_buckets['avg']
            df_buckets['std'] = np.sqrt(df_buckets['var'])
            df_buckets['std_tcl'] = df_buckets['std'] / np.sqrt(df_buckets['cnt'])

            print(df_buckets)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(df_buckets['x'],
                        df_buckets['avg']*1e4,
                        yerr=df_buckets['std_tcl']*1e4,
                        fmt='o', capsize=5,
                        label='Bucket Mean bps ± CI')
            ax.grid()
            ax.set_xlabel('Qtl X')
            ax.set_ylabel('Y')
            ax.set_title(f'Bucket Plot: {self.name}')
            ax.legend()
            try_to_save_png(self.save_graph_path)

        return results_mean, results_std
