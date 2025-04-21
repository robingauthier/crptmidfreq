
import pandas as pd

from crptmidfreq.outlier.hampel import hampel_ts

# pytest ./crptmidfreq/outlier/tests/test_hampel.py --pdb --maxfail=1


def test_hampel_ts():
    serie = pd.Series([1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       - 2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       - 2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1000, -1000,  # the jump
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8])
    df = pd.DataFrame({'in': serie.cumsum()})
    df['dscode'] = 'A'
    df['date'] = 1
    df['date'] = df['date'].cumsum()
    df['date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df['date'], unit='D')
    df = df.set_index(['dscode', 'date'])
    nb_out_1 = hampel_ts(df, ['in'],
                         name='test_hampel',
                         window=10,
                         mincnt=10)
    assert nb_out_1.shape[0] == 1
    assert nb_out_1['nb_out'].sum() == 1
    df = pd.DataFrame({'in': serie})
    df['dscode'] = 'A'
    df['date'] = 1
    df['date'] = df['date'].cumsum()
    df['date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df['date'], unit='D')
    df = df.set_index(['dscode', 'date'])
    nb_out_2 = hampel_ts(df, ['in'],
                         name='test_hampel',
                         window=10,
                         mincnt=10)
    assert nb_out_2['nb_out'].sum() == 0


def test_hampel_ts2():
    serie = pd.Series([1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       - 2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       - 2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1000, -1000,  # the jump
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       - 2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       - 2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       -1000, 1000,  # the jump
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,
                       1, 2, 3, -2, -3, 4, -5, 6, 3, -5, 6, 10, -8,

                       ])
    df = pd.DataFrame({'in': serie.cumsum()})
    df['dscode'] = 'A'
    df['date'] = 1
    df['date'] = df['date'].cumsum()
    df['date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df['date'], unit='D')
    df = df.set_index(['dscode', 'date'])
    nb_out_1 = hampel_ts(df, ['in'],
                         name='test_hampel',
                         window=10,
                         mincnt=10)
    assert nb_out_1.shape[0] == 2
    assert nb_out_1['nb_out'].sum() == 2
