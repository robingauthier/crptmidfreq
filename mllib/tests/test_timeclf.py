
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from pandas.tseries.offsets import BDay
from joblib import delayed, Parallel
from functools import partial

from crptmidfreq.utils.common import lvals, filter_date_df
from crptmidfreq.mllib.timeclf import TimeSplitClf

# pytest ./crptmidfreq/mllib/tests/test_timeclf.py --pdb --maxfail=1


def test_timeclf():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # ----------------------------
    # 1. Fake Data Generation
    # ----------------------------
    np.random.seed(42)

    # Generate fake dates
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="B")  # Business days

    # Generate fake dscode identifiers
    dscodes = ['A', 'B', 'C', 'D', 'E']
    index = pd.MultiIndex.from_product([dscodes, dates], names=['dscode', 'date'])

    # Generate fake features and target
    df = pd.DataFrame(index=index)
    df["feature1"] = np.random.randn(len(df))
    df["feature2"] = np.random.randn(len(df))
    df["y"] = df["feature1"] * 2 + df["feature2"] * 0.5 + np.random.randn(len(df))
    df["w"] = np.random.rand(len(df))  # Random sample weights

    # ----------------------------
    # 2. Define a simple model function
    # ----------------------------
    def simple_model():
        return LinearRegression()

    # ----------------------------
    # 3. Initialize and Run TimeSplitClf
    # ----------------------------
    clf = TimeSplitClf(fct=simple_model)

    # Extracting X, y, and weights
    X = df.drop(columns=["y", "w"])
    y = df["y"]
    w = df["w"]

    # Running fit_predict
    predictions = clf.fit_predict(X, y, w)
    df['ypred'] = predictions

    assert df[['y', 'ypred']].corr().iloc[0, 1] > 0.7
