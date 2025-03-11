import numpy as np

from utils.common import clean_folder
from ..incr_orderbook_imb2 import OrderBookImb1Stepper


def test_imb_values_basic():
    dscode = np.int64(np.ones(20))
    dt = np.int64(np.arange(20))
    bidask = np.array(['a'] * 10 + ['b'] * 10)
    price = np.int64(np.linspace(100, 120, 20))
    quantity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

    m = OrderBookImb1Stepper()
    # dt, dscode, bidask, price, quantity
    rb, ra = m.update(dt, dscode, bidask, price, quantity)
    ea = np.array([1., 2., 2., 3., 3., 4., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.])
    eb = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   1., 1., 2., 2., 3., 4., 5., 6., 7., 7.])
    np.testing.assert_array_equal(ra, ea)
    np.testing.assert_array_equal(rb, eb)


def test_imb_values_basic_load_save():
    dscode = np.int64(np.ones(20))
    dt = np.int64(np.arange(20))
    bidask = np.array(['a'] * 10 + ['b'] * 10)
    price = np.int64(np.linspace(100, 120, 20))
    quantity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    c = 15
    clean_folder('test_orderbook')
    m1 = OrderBookImb1Stepper.load(folder='test_orderbook', name='')
    rb1, ra1 = m1.update(dt[:c], dscode[:c], bidask[:c], price[:c], quantity[:c])
    m2 = OrderBookImb1Stepper.load(folder='test_orderbook', name='')
    rb2, ra2 = m2.update(dt[c:], dscode[c:], bidask[c:], price[c:], quantity[c:])
    ra = np.concatenate([ra1, ra2])
    rb = np.concatenate([rb1, rb2])

    ea = np.array([1., 2., 2., 3., 3., 4., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.])
    eb = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   1., 1., 2., 2., 3., 4., 5., 6., 7., 7.])
    np.testing.assert_array_equal(ra, ea)
    np.testing.assert_array_equal(rb, eb)
