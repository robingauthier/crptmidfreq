import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


@njit
def ridge_fit(x, y, lam):
    """
    Perform Ridge regression: y = alpha + beta * x + residual

    Args:
        x: Independent variable (1D array)
        y: Dependent variable (1D array)
        lam: Regularization parameter (lambda)

    Returns:
        alpha: Intercept of the regression
        beta: Slope of the regression
        residuals: Residuals of the regression
    """

    # Compute means
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    # Center x and y
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Compute beta (slope) with regularization
    x_var = np.nansum(x_centered ** 2)
    xy_cov = np.nansum(x_centered * y_centered)
    if (x_var + lam) > 0:
        beta = xy_cov / (x_var + lam)
    else:
        beta = 0.0

    # Compute alpha (intercept)
    alpha = y_mean - beta * x_mean

    # Compute residuals
    y_pred = alpha + beta * x
    residuals = y - y_pred

    return alpha, beta, residuals


@njit
def update_rolling_reg(timestamps,
                       dscode,
                       values1,
                       values2,
                       position,
                       rolling_dict1,
                       rolling_dict2,
                       last_timestamps,
                       window,
                       lam):
    """

    """
    result_alpha = np.zeros(len(dscode), dtype=np.float64)
    result_beta = np.zeros(len(dscode), dtype=np.float64)
    result_resid = np.zeros(len(dscode), dtype=np.float64)
    for i in range(len(dscode)):
        code = dscode[i]
        value1 = values1[i]
        value2 = values2[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        if code not in position:
            position[code] = 0

        if code not in rolling_dict1:
            rolling_dict1[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_dict1[code][j] = np.nan
        if code not in rolling_dict2:
            rolling_dict2[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_dict2[code][j] = np.nan

        position_loc = position[code]
        rolling_dict1[code][position_loc] = value1
        rolling_dict2[code][position_loc] = value2
        last_timestamps[code] = ts
        position[code] = (position_loc + 1) % window

        alpha, beta, residuals = \
            ridge_fit(rolling_dict1[code], rolling_dict2[code], lam)
        result_alpha[i] = alpha
        result_beta[i] = beta
        result_resid[i] = residuals[position_loc]
    return result_alpha, result_beta, result_resid


class RollingRidgeStepper(BaseStepper):
    def __init__(self, folder='', name='', window=1, lam=0.):
        super().__init__(folder,name)

        self.window = window
        self.lam = lam
        self.position = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.values1 = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.values2 = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_ts = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name='', window=1,lam=0.1):
        """Load instance from saved state or create new if not exists"""
        return RollingRidgeStepper.load_utility(cls,folder=folder,name=name,
                                                window=window,lam=lam)


    def update(self, dt, dscode, values1, values2):
        self.validate_input(dt,dscode,values1,serie2=values2)
        alpha, beta, resid = update_rolling_reg(dt.view(np.int64), dscode, values1, values2,
                                                self.position, self.values1, self.values2,
                                                self.last_ts, self.window, self.lam)
        return alpha, beta, resid
