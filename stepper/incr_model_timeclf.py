import pandas as pd
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger

# Quite similar to the featurelib/timeclf.py but it is a stepper

logger = get_logger()


def get_dts_max_before(dts, train_start_dt):
    resp = np.where(dts < train_start_dt)[0]
    if resp.shape[0] > 0:
        return resp.max()+1
    return 0


def get_dts_min_after(dts, train_end_dt):
    resp = np.where(dts > train_end_dt)[0]
    if resp.shape[0] > 0:
        return resp.min()-1
    return dts.shape[0]


class TimeClfStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='', lookback=300, minlookback=100,
                 fitfreq=10, gap=1):
        """
        """
        super().__init__(folder, name)
        self.fitfreq = fitfreq
        self.gap = gap
        self.lookback = lookback  # in time units
        self.minlookback = minlookback  # in time units

        # History of train/test dates
        self.time_unit = None
        self.ltimes = []

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, minlookback=100,
             fitfreq=10, gap=1, model_gen=None, with_fit=True):
        """Load instance from saved state or create new if not exists"""
        return TimeClfStepper.load_utility(cls,
                                           folder=folder,
                                           name=name,
                                           lookback=lookback,
                                           minlookback=minlookback,
                                           fitfreq=fitfreq,
                                           gap=gap,
                                           )

    def update(self, dts, dscode, serie):
        """
        Rolling train/predict update.

        Parameters:
            dts   : list or array of dates corresponding to each row in `serie`
            serie : numpy array of shape (n_samples, n_features)

        Returns:
            result : numpy array of predictions (shape: n_samples,)
        """
        assert np.all(np.diff(dts) >= 0)
        # Compute a representative time unit if not already set.
        if self.time_unit is None:
            dts_arr = np.array(dts)
            time_diffs = np.diff(dts_arr)
            positive_diffs = time_diffs[time_diffs > 0]
            self.time_unit = np.median(positive_diffs) if len(positive_diffs) > 0 else 1

        n = serie.shape[0]
        result = np.zeros(n, dtype=np.float64)

        first_date_dt = dts[0]
        last_date_dt = dts[-1]

        if len(self.ltimes) == 0:
            train_end_dt = first_date_dt + self.minlookback * self.time_unit
        else:
            train_end_dt = self.ltimes[-1]['train_stop_dt']
            # Incrementing train_end_dt
            train_end_dt = train_end_dt+self.fitfreq*self.time_unit

        while train_end_dt < last_date_dt:
            # we work on train_end_dt

            # Define the training window: use the last 'lookback' samples (or fewer if at the start).
            train_start_dt = max(0, train_end_dt - self.lookback * self.time_unit)
            train_start = get_dts_max_before(dts, train_start_dt)
            train_start = max(0, train_start)
            train_end = get_dts_max_before(dts, train_end_dt)

            # Define the test window:
            test_start_dt = train_end_dt+self.gap*self.time_unit
            test_start = get_dts_min_after(dts, test_start_dt)
            test_end_dt = test_start_dt+self.fitfreq*self.time_unit
            test_end_dt = min(test_end_dt, last_date_dt)
            test_end = get_dts_min_after(dts, test_end_dt)

            train_start_dt_str = pd.to_datetime(train_start_dt*1e3).strftime('%Y-%m-%d')
            train_end_dt_str = pd.to_datetime(train_end_dt*1e3).strftime('%Y-%m-%d')
            test_start_dt_str = pd.to_datetime(test_start_dt*1e3).strftime('%Y-%m-%d')
            test_end_dt_str = pd.to_datetime(test_end_dt*1e3).strftime('%Y-%m-%d')

            strloc = (f'Train on [{train_start_dt_str} - {train_end_dt_str}] ' +
                      f'Predict on [{test_start_dt_str} - {test_end_dt_str}] ')
            self.ltimes += [{
                'train_start_dt': train_start_dt,
                'train_stop_dt': train_end_dt,

                'train_start_dtn': test_start_dt_str,
                'train_stop_dtn': train_end_dt_str,

                'train_start_i': train_start,
                'train_stop_i': train_end,

                'test_start_dt': test_start_dt,
                'test_stop_dt': test_end_dt,

                'test_start_dtn': test_start_dt_str,
                'test_stop_dtn': test_end_dt_str,

                'test_start_i': test_start,
                'test_stop_i': test_end,

                'str': strloc}]

            # Incrementing train_end_dt
            train_end_dt = train_end_dt+self.fitfreq*self.time_unit

        return result
