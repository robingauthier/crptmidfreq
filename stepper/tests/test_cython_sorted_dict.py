import numpy as np
import pandas as pd

from stepper.cython_sorted_dict import MultiSortedDict
from stepper.cython_sorted_dict import MySortedDict


def test_mysorteddict_ask():
    # cur_min is a min with order=1
    # hence we need ascending order
    d = MySortedDict()
    d.add(100, 0.1)
    d.add(99, 0.5)
    d.add(94, 0.8)
    d.add(95, 0.8)
    d.add(290, 1.0)
    d.add(100000, 1.0)
    d.pop(95)
    df = pd.DataFrame({'k': d.get_keys(), 'v': d.get_values()})
    ek = np.array([94, 99, 100, 290])
    ev = np.array([0.8, 0.5, 0.1, 1.0])
    np.testing.assert_array_equal(df['k'].values, ek)
    np.testing.assert_array_equal(df['v'].values, ev)


def test_mysorteddict_bid():
    # cur_min is a max with order=0
    # hence we should first see the max
    # order must be decreasing
    d = MySortedDict(order=0)
    d.add(100, 0.1)
    d.add(99, 0.5)
    d.add(94, 0.8)
    d.add(95, 0.8)
    d.add(290, 1.0)
    d.pop(95)
    df = pd.DataFrame({'k': d.get_keys(), 'v': d.get_values()})
    ek = np.array([290, 100, 99, 94])
    ev = np.array([1.0, 0.1, 0.5, 0.8])
    np.testing.assert_array_equal(df['k'].values, ek)
    np.testing.assert_array_equal(df['v'].values, ev)


def test_multi_sorteddict_ask():
    # cur_min is a min with order=1
    # hence we need ascending order
    d = MultiSortedDict()
    d.add(1, 100, 0.1)
    d.add(2, 100, 0.1)
    d.add(2, 95, 0.1)
    d.pop(2, 100)
    d.add(1, 99, 0.5)
    d.add(1, 94, 0.8)
    d.add(1, 95, 0.8)
    d.add(1, 290, 1.0)
    d.add(1, 100000, 1.0)
    d.pop(1, 95)
    df = pd.DataFrame({'k': d.get_keys(1), 'v': d.get_values(1)})
    ek = np.array([94, 99, 100, 290])
    ev = np.array([0.8, 0.5, 0.1, 1.0])
    np.testing.assert_array_equal(df['k'].values, ek)
    np.testing.assert_array_equal(df['v'].values, ev)


def test_multi_sorteddict_complex():
    d = MultiSortedDict()
    for i in range(100):
        d.add(1, 1000 + i, i / 100)
    df = pd.DataFrame({'k': d.get_keys(1), 'v': d.get_values(1)})
    assert df.shape[0] == 99
    assert df.iloc[0, 0] == 1001
    for i in range(100):
        d.add(1, 1000 + i, 0)
    # now we need to switch up
    for i in range(100):
        d.add(1, 5000 + i, i / 100)
    df = pd.DataFrame({'k': d.get_keys(1), 'v': d.get_values(1)})
    assert df.shape[0] == 99
    assert df.iloc[0, 0] == 5001


def test_multi_sorteddict_complex2():
    d = MultiSortedDict(order=0)
    for i in range(100):
        d.add(1, 5000 + i, i / 100)
    df = pd.DataFrame({'k': d.get_keys(1), 'v': d.get_values(1)})
    assert df.shape[0] == 99
    assert df.iloc[0, 0] == 5099
    for i in range(100):
        d.add(1, 5000 + i, 0)
    # now we need to switch up
    for i in range(100):
        d.add(1, 1000 + i, i / 100)
    df = pd.DataFrame({'k': d.get_keys(1), 'v': d.get_values(1)})
    assert df.shape[0] == 99
    assert df.iloc[0, 0] == 1099
    for i in range(100):
        d.add(1, 1000 + i, 0)
    for i in range(100):
        d.add(1, i, i / 100)
    df = pd.DataFrame({'k': d.get_keys(1), 'v': d.get_values(1)})
    assert df.shape[0] == 99
    assert df.iloc[0, 0] == 99
