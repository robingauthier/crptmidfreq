
import pandas as pd
from sklearn.datasets import fetch_openml

from crptmidfreq.outlier.hampel import hampel_ts

# pytest ./crptmidfreq/outlier/tests/test_hampel_ecg.py --pdb --maxfail=1


def get_ecg_data():
    # ECG5000 is a univariate series classification set (with some outlier beats)
    ecg = fetch_openml(name="ECG5000", as_frame=False)
    X, y = ecg["data"], ecg["target"]
    ldf = []
    for i in range(X.shape[1]):
        dfloc = pd.DataFrame({'serie': X[:, i]})
        dfloc['seriec'] = dfloc['serie'].cumsum()
        dfloc['dscode'] = str(i)
        dfloc['date'] = 1
        dfloc['date'] = dfloc['date'].cumsum()
        dfloc['date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(dfloc['date'], unit='D')
        ldf += [dfloc]
    df = pd.concat(ldf, axis=0, sort=False)
    df = df.set_index(['dscode', 'date'])
    #df[df['dscode'] == '0']['seriec'].plot()
    return df


def test_hampel_ecg():
    df = get_ecg_data()
    nb_out = hampel_ts(df)
    # there are no outliers here really
