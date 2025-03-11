import numpy as np

from stepper.incr_orderbook_bbo2 import OrderBookBboStepper
from utils.common import clean_folder


def test_ffill_values_basic():
    dscode = np.int64(np.ones(20))
    dt = np.int64(np.arange(20))
    bidask = np.array(['a'] * 10 + ['b'] * 10)
    price = np.int64(np.linspace(100, 120, 20))
    quantity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

    m = OrderBookBboStepper()
    rb, ra = m.update(dt, dscode, bidask, price, quantity)
    eb = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   110., 110., 112., 112., 114., 115., 116., 117., 118., 118.])
    np.testing.assert_array_equal(rb, eb)


def test_ffill_values_basic_load_save():
    dscode = np.int64(np.ones(20))
    dt = np.int64(np.arange(20))
    bidask = np.array(['a'] * 10 + ['b'] * 10)
    price = np.int64(np.linspace(100, 120, 20))
    quantity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    c = 15
    clean_folder('test_orderbook')
    m1 = OrderBookBboStepper.load(folder='test_orderbook', name='')
    rb1, ra1 = m1.update(dt[:c], dscode[:c], bidask[:c], price[:c], quantity[:c])
    m2 = OrderBookBboStepper.load(folder='test_orderbook', name='')
    rb2, ra2 = m2.update(dt[c:], dscode[c:], bidask[c:], price[c:], quantity[c:])
    rb = np.concatenate([rb1, rb2])
    eb = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                   110., 110., 112., 112., 114., 115., 116., 117., 118., 118.])
    np.testing.assert_array_equal(rb, eb)
