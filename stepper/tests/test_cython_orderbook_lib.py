import numpy as np
from stepper.cython_orderbook_lib import lambda_imb1
from stepper.cython_orderbook_lib import lambda_level1
from stepper.cython_orderbook_lib import update_orderbook_level1
from stepper.cython_sorted_dict import MultiSortedDict


def test_bbo():
    dscode = np.int64(np.ones(20))
    dt = np.int64(np.arange(20))
    bidask = np.array(['a'] * 10 + ['b'] * 10)
    price = np.int64(np.linspace(100, 120, 20))
    quantity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    last_bid = MultiSortedDict(order=1)
    last_ask = MultiSortedDict(order=0)
    last_ts = {}
    rb, ra = update_orderbook_level1(dscode, bidask, price, quantity, dt,
                                     last_bid, last_ask, last_ts, lambda_level1)
    eb = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   110., 110., 112., 112., 114., 115., 116., 117., 118., 118.])
    np.testing.assert_array_equal(rb, eb)


def test_imb():
    dscode = np.int64(np.ones(20))
    dt = np.int64(np.arange(20))
    bidask = np.array(['a'] * 10 + ['b'] * 10)
    price = np.int64(np.linspace(100, 120, 20))
    quantity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    last_bid = MultiSortedDict(order=1)
    last_ask = MultiSortedDict(order=0)
    last_ts = {}
    rb, ra = update_orderbook_level1(dscode, bidask, price, quantity, dt,
                                     last_bid, last_ask, last_ts, lambda_imb1)
    ea = np.array([1., 2., 2., 3., 3., 4., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.])
    eb = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   1., 1., 2., 2., 3., 4., 5., 6., 7., 7.])
    np.testing.assert_array_equal(ra, ea)
    np.testing.assert_array_equal(rb, eb)
