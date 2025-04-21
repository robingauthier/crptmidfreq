
import numpy as np
from numba import njit

from crptmidfreq.stepper.rolling_base import RollingStepper


@njit(cache=True)
def update_rolling_theta(timestamps,
                        dscode,
                        values,
                        position,
                        rolling_dict,
                        last_timestamps,
                        window,
                        theta):
    """
    https://nixtlaverse.nixtla.io/statsforecast/docs/models/standardtheta.html
        
    The Theta lines are estimated by modifying the ‘curvatures’ of the original time series, 
    i.e., the distances between the points of the series, with those of a simple linear regression line

    $$
    \zeta_t^{(\theta)} = \theta Y_t + (1 - \theta)(A_T + B_T t), \quad \text{for } t = 1, \dots, T
    $$
    $$
    B_T = \frac{6}{T(T^2 - 1)} \left( 2 \sum_{t=1}^{T} t Y_t - (T + 1) \sum_{t=1}^{T} Y_t \right)
    $$
    $$
    A_T = \frac{1}{T} \sum_{t=1}^{T} Y_t - \frac{T + 1}{2} B_T
    $$


    """
    result = np.zeros(len(dscode), dtype=np.float64)
    for i in range(len(dscode)):
        code = dscode[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        if code not in position:
            position[code] = 0

        if code not in rolling_dict:
            rolling_dict[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_dict[code][j] = np.nan

        position_loc = position[code]
        rolling_dict[code][position_loc] = value
        last_timestamps[code] = ts
        position[code] = (position_loc + 1) % window
    
        # computing Sum t*Y_t
        arr_t = (np.arange(window)+window-position_loc)%window
        y_x_t = rolling_dict[code]*arr_t

        b_t=6/(window*(window**2-1)) * (2*np.nansum(y_x_t) - (window+1)*np.nansum(rolling_dict[code]))
        a_t = 1/window*np.nansum(rolling_dict[code])- (window+1)/2 * b_t
        result[i] = theta*value + (1 - theta) * (a_t + b_t * window)
        #result[i] = value
        
        
    return result


class ThetaStepper(RollingStepper):
    def __init__(self, folder='', name='', window=1,theta=2.0):
        super().__init__(folder, name,window)
        self.theta=theta
        
    def update(self, dt, dscode, values):
        self.validate_input(dt, dscode, values)
        res = update_rolling_theta(dt.view(np.int64), dscode, values,
                                  self.position, self.values, self.last_ts, self.window,self.theta)
        return res
    @classmethod
    def load(cls, folder, name, window=1,theta=2.0):
        """Load instance from saved state or create new if not exists"""
        cls= RollingStepper.load_utility(cls, folder=folder, name=name, window=window)
        cls.theta=theta
        return cls
