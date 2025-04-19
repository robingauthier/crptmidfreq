from sklearn.base import RegressorMixin, TransformerMixin


class LocalModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_generator=None, by_col='dscode'):
        self.models = {}
        self.model_generator = model_generator
        self.by_col = by_col

    def fit(self, df, target_col, wgt_col='one'):
        pass

    def predict(self, df):
        lr = []
        for id, dfg in df.groupby(self.by_col):
            if id in self.models:
                pred = self.models[id].predict(dfg)
                lr.append(pred)
        import pdb
        pdb.set_trace()
        return self.model.predict(data)
