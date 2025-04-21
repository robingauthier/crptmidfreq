import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class LocalModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_generator=None, by_col='dscode'):
        self.models = {}
        self.model_generator = model_generator
        self.by_col = by_col
        self.fit_colname_syntax = fit_colname_syntax

    def fit(self, X, y, wgt):
        assert np.all(X.index == y.index)
        assert np.all(X.index == wgt.index)
        assert X.index.duplicated().sum() == 0
        by_col = self.by_col
        by_vals = X[by_col].unique()
        for by_val in by_vals:
            indices = X[X[by_col] == by_val].index
            model_loc = self.model_generator()
            model_loc.fit(X.loc[indices], y.loc[indices], wgt.loc[indices])
            self.models[by_val] = model_loc

    def predict(self, df):
        lr = []
        for id, dfg in df.groupby(self.by_col):
            if id in self.models:
                pred = self.models[id].predict(dfg)
                lr.append(pred)
        rdf = pd.concat(lr)
        return rdf


class LocalModel2(BaseEstimator, RegressorMixin):
    """fit syntax is using col names"""

    def __init__(self, model_generator=None, by_col='dscode'):
        self.models = {}
        self.model_generator = model_generator
        self.by_col = by_col

    def fit(self, df, target_col='', wgt_col='one'):
        by_col = self.by_col
        by_vals = df[by_col].unique()
        for by_val in by_vals:
            dfloc = df[df[by_col] == by_val]
            model_loc = self.model_generator()
            model_loc.fit(dfloc, target_col=target_col, wgt_col=wgt_col)
            self.models[by_val] = model_loc

    def predict(self, df):
        lr = []
        for id, dfg in df.groupby(self.by_col):
            if id in self.models:
                pred = self.models[id].predict(dfg)
                pred = pd.Series(pred, index=dfg.index)
                lr.append(pred)
        rdf = pd.concat(lr)
        return rdf
