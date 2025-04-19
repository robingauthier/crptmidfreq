
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from pandas.tseries.offsets import BDay
from joblib import delayed, Parallel
from functools import partial

from crptmidfreq.utils.common import lvals, filter_date_df

# This is the rolling train / predict class


def dts_split(dts, train_freq=250, min_train_size=400, max_train_days=600, train_gap=1):
    """
    dts : list of dates
    dts_df = dts.to_frame('date').drop_duplicates().assign(one=1).set_index('date').sort_index()['date'].tolist()
    """
    assert isinstance(dts, list)
    dts = sorted(dts)
    if len(dts) <= min_train_size:
        n0 = len(dts)
        n1 = int(n0 * 0.7)
        print(f'dts_split asked to split but we have less than min_train_size days {n0}')
        train_start = dts[0]
        train_stop = dts[n1]
        test_start = dts[n1 + 1]
        test_stop = dts[-1]
        return [{'train_start': train_start,
                 'train_stop': train_stop,
                 'test_start': test_start,
                 'test_stop': test_stop,
                 'str': 'warning dts'}]
    lr = []
    n = len(dts)
    stop_iter = min_train_size
    while stop_iter + train_gap < n:
        train_stop = dts[stop_iter]
        train_start = dts[max(0, stop_iter - max_train_days)]
        test_start = dts[stop_iter + train_gap]
        is_last_iter = (stop_iter + train_freq >= n)
        if is_last_iter:
            # to allow for new data
            test_stop = dts[-1] + BDay(50)
        else:
            test_stop = dts[stop_iter + train_freq + train_gap - 1]
        stop_iter += train_freq
        ntrain_start = train_start.strftime('%Y-%m-%d')
        ntrain_stop = train_stop.strftime('%Y-%m-%d')
        ntest_start = test_start.strftime('%Y-%m-%d')
        ntest_stop = test_stop.strftime('%Y-%m-%d')
        strloc = f'Train on [{ntrain_start} - {ntrain_stop}] Predict on [{ntest_start} - {ntest_stop}] '
        lr += [{'train_start': train_start,
                'train_stop': train_stop,
                'test_start': test_start,
                'test_stop': test_stop,
                'str': strloc}]
    return lr

# the time TimeSplitClf will operate on pandas dataframes


def run_fit_predict_loc(train_df, test_df, dtd, fct=None, is_transform=False):
    """
    Internal function used in the TimeSplitClf class


    dtd : the date dictionary with train_stop as key
    """
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    print('run_fit_predict_loc in tsclf 2')
    if fct is None:
        print('ts_rolling_pipeline_v3 :: self.fct() is None')
        return {'pipe': None, 'rdf': None, 'train_stop': dtd['train_stop']}
    if train_df.drop(['todel_y', 'todel_w'], axis=1).dropna(how='all', axis=0).shape[0] == 0:
        print('ts_rolling_pipeline_v3 :: train_df is nan')
        return {'pipe': None, 'rdf': None, 'train_stop': dtd['train_stop']}

    # calling fct will create a model that is empty/non fitted
    pipeloc = fct()
    pipeloc.fit(train_df.drop(['todel_y', 'todel_w'], axis=1),
                train_df['todel_y'],
                train_df['todel_w'])
    if is_transform:
        rdfloc = pipeloc.transform(test_df.drop(['todel_y', 'todel_w'], axis=1))
    else:
        rdfloc = pipeloc.predict(test_df.drop(['todel_y', 'todel_w'], axis=1))

    if (not isinstance(rdfloc, pd.Series)) and (not isinstance(rdfloc, pd.DataFrame)):
        rdfloc = pd.Series(rdfloc, index=test_df.index)
    return {'rdf': rdfloc,
            'pipe': pipeloc,
            'train_stop': dtd['train_stop']}


class TimeSplitClf(BaseEstimator, RegressorMixin):
    """lot of copy paste from utils/clf.py
    Sa for my implementation of this class.
    """

    def __init__(self, fct,
                 # Split arguments
                 train_freq=250,
                 min_train_size=400,
                 max_train_days=600,
                 train_gap=1,
                 train_remove_covid=True,
                 n_jobs=0,
                 # other arguments
                 verbose=1,
                 extraindex=[],
                 is_transform=False):
        self.fct = fct
        self.train_freq = train_freq
        self.train_remove_covid = train_remove_covid
        self.min_train_size = min_train_size
        self.max_train_days = max_train_days
        self.train_gap = train_gap
        # n_jobs
        self.n_jobs = n_jobs
        # other parameters
        self.verbose = verbose
        self.extraindex = extraindex
        self.is_transform = is_transform

    def fit_predict(self, X, y, w):
        """Easier to do the predict also in one go"""
        assert isinstance(X.index, pd.MultiIndex)
        assert X.index.names == ['dscode', 'date'] + self.extraindex
        assert X.index.duplicated().sum() == 0
        assert 'todel_y' not in X.columns
        assert 'todel_w' not in X.columns
        dts = lvals(X, 'date')
        dts = dts.to_frame('date')\
            .drop_duplicates()['date'].tolist()
        ldt = dts_split(dts,
                        train_freq=self.train_freq,
                        min_train_size=self.min_train_size,
                        max_train_days=self.max_train_days,
                        train_gap=self.train_gap)
        self.ldt = ldt

        # Creating the argument list for the run
        df = pd.concat([X, y.to_frame('todel_y'), w.to_frame('todel_w')], axis=1)
        largs = []
        for dtd in ldt:
            train_df = filter_date_df(df, start_date=dtd['train_start'], end_date=dtd['train_stop'])
            test_df = filter_date_df(df, start_date=dtd['test_start'], end_date=dtd['test_stop'])
            if test_df.shape[0] == 0 or train_df.shape[0] == 0:
                continue
            largs += [{'train_df': train_df, 'test_df': test_df, 'dtd': dtd}]

        prun_fit_predict_loc = partial(run_fit_predict_loc, fct=self.fct, is_transform=self.is_transform)
        # Running in parrallel or not
        if self.n_jobs > 0:
            tasks = (delayed(prun_fit_predict_loc)(**args) for args in largs)
            results = Parallel(n_jobs=self.n_jobs)(tasks)
        else:
            results = []
            for arg in largs:
                results += [prun_fit_predict_loc(**arg)]
        dmodels = {}
        for result in results:
            dmodels[result['train_stop']] = result['pipe']
        lr = []
        for result in results:
            lr += [result['rdf']]
        rdf = pd.concat(lr, axis=0)
        train_stop = min([x['dtd']['train_stop'] for x in largs])
        assert rdf.index.duplicated().sum() == 0
        rdf = rdf.sort_index(level=['date', 'dscode'])

        # check on the length
        n0 = X.loc[lambda x: lvals(x, 'date') > train_stop].shape[0]
        n1 = rdf.shape[0]
        if n1 != n0:
            print(
                f'ts_rolling_pipeline_v3 initial dataset had {n0} rows and now post pipeline we have {n1} rows -> '
                f'{n1 / n0}')
        nX = X.loc[lambda x: lvals(x, 'date') > train_stop]
        n0 = rdf.shape[0] / 1e6
        n1 = nX.shape[0] / 1e6
        if n1 != n0:
            print(
                f'ts_rolling_pipeline_v3 initial dataset had {n0} rows and now post pipeline we have {n1} rows -> '
                f'{n1 / n0}')
        if n1 == n0:
            assert np.all(sorted(nX.index) == sorted(rdf.index))
        self.dmodels = dmodels
        return rdf

    def predict(self, X):
        assert isinstance(X.index, pd.MultiIndex)
        assert X.index.names == ['dscode', 'date'] + self.extraindex
        assert X.index.duplicated().sum() == 0
        assert len(self.ldt) > 0
        assert len(self.ldt) == len(self.dmodels)
        lr = []
        for dtd in self.ldt:
            X_loc = filter_date_df(X, start_date=dtd['test_start'], end_date=dtd['test_stop'])
            pipeloc = self.dmodels[dtd['train_stop']]
            rdfloc = pipeloc.predict(X_loc)
            if not isinstance(rdfloc, pd.Series):
                rdfloc = pd.Series(rdfloc, index=X_loc.index)
            lr += [rdfloc]
        rdf = pd.concat(lr, axis=0)
        assert rdf.index.duplicated().sum() == 0
        return rdf


# ipython -i -m featurelib.timeclf
if __name__ == '__main__':
    import pandas as pd
    dts = pd.date_range('2010-01-01', '2020-01-01').tolist()
    l1 = dts_split(dts)

    example_TimeSplit()
