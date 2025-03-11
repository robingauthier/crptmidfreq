import os
import pickle

import numpy as np
from numba import int_, float_
from numba import njit
from numba import types
from numba.typed import Dict

from config_loc import get_data_folder


@njit(float_(int_, float_, float_, float_))
def create_continuous_smooth_kernel(t, spike_interval_ms, half_life_ms, spike_width_ms):
    """
    Create a continuous smooth kernel with spikes that have a Gaussian profile.

    """
    assert t >= 0
    assert spike_interval_ms > 0
    assert spike_width_ms > 0
    assert half_life_ms > 0
    tr = t - (t // spike_interval_ms) * spike_interval_ms

    spike_term = np.exp(-0.5 * (tr / spike_width_ms) ** 2)

    # Apply the EWM envelope
    alpha = np.log(2) / half_life_ms
    ewm_weights = np.exp(-alpha * t)
    res = ewm_weights * spike_term
    return res


@njit
def update_rolling_kernel_twap(timestamps,
                               dscode,
                               values,
                               units,
                               position,  # position in the memory
                               rolling_dict,  # memory
                               rolling_units_dict,
                               last_timestamps,
                               window,
                               spike_interval_ms,
                               half_life_ms,
                               spike_width_ms,
                               ):
    """

    """

    result = np.zeros(len(dscode), dtype=np.float64)

    for i in range(len(dscode)):
        code = dscode[i]
        value = values[i]
        unit = units[i]
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
            rolling_units_dict[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_units_dict[code][j] = np.nan

        position_loc = position[code]
        rolling_dict[code][position_loc] = value
        rolling_units_dict[code][position_loc] = unit
        last_timestamps[code] = ts

        ### unit is time or cumulative volume
        ## kernel has to be a continuous function
        resloc = 0
        for l in range(window):
            l_loc = (position_loc - l) % window
            if np.isnan(rolling_units_dict[code][l_loc]):
                break
            t_loc = unit - rolling_units_dict[code][l_loc]
            kernel = create_continuous_smooth_kernel(t_loc,
                                                     spike_interval_ms,
                                                     half_life_ms,
                                                     spike_width_ms)
            # you could also do a product here ?
            resloc += kernel * rolling_dict[code][l_loc]
        position[code] = (position_loc + 1) % window
        result[i] = resloc
    return result


class RollingKernelTwapStepper:
    """
    copy paste from RollingStepper
    Convolution with a kernel that respects time or volume
    and looks like this  1 0 0 0 1 0 0 0 1 0 0 0
    Idea was to be able to detect vwap or twaps
    """

    def __init__(self, folder='', name='',
                 window=100,
                 spike_interval=10.0,
                 half_life=20.0,
                 spike_width=2.0):
        self.folder = folder
        self.name = name
        self.window = window
        self.spike_interval = spike_interval
        self.half_life = half_life
        self.spike_width = spike_width
        self.values = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.units = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_ts = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.position = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        """Save internal state to file"""
        folder_path = os.path.join(get_data_folder(), self.folder)
        filepath = os.path.join(folder_path, self.name + '.pkl')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        state = {
            'last_ts': {k: v for k, v in self.last_ts.items()},
            'position': {k: v for k, v in self.position.items()},
            'values': {k: list(v) for k, v in self.values.items()},
            'units': {k: list(v) for k, v in self.units.items()},
            'window': self.window,
            'spike_interval': self.spike_interval,
            'half_life': self.half_life,
            'spike_width': self.spike_width,

        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder='', name='',
             window=None,
             spike_interval=None,
             half_life=None,
             spike_width=None
             ):
        """Load instance from saved state or create new if not exists"""
        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            print(f'RollingStepper creating instance {folder} {name} {window}')
            return cls(folder=folder, name=name, window=window, half_life=half_life,
                       spike_interval=spike_interval, spike_width=spike_width)

        print(f'RollingStepper loading instance {folder} {name}')
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        instance = cls(folder=folder, name=name)
        instance.folder = folder
        instance.name = name
        instance.window = state['window']
        instance.spike_interval = state['spike_interval']
        instance.half_life = state['half_life']
        instance.spike_width = state['spike_width']
        instance.values = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        instance.units = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        instance.last_ts = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        instance.position = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        for k, v in state['values'].items():
            instance.values[k] = np.array(v)
        for k, v in state['units'].items():
            instance.units[k] = np.array(v)
        for k, v in state['last_ts'].items():
            instance.last_ts[k] = v
        for k, v in state['position'].items():
            instance.position[k] = v
        return instance

    def update(self, dt, dscode, values, units):
        if len(dscode) != len(values):
            raise ValueError("Codes and values arrays must have the same length")
        if len(dt) != len(values):
            raise ValueError("Codes and values arrays must have the same length")
        if len(dt) != len(units):
            raise ValueError("Codes and values arrays must have the same length")

        if not dt.dtype == 'int64':
            timestamps = dt.astype('datetime64[ns]').astype('int64')
        else:
            timestamps = dt
        print('started running update_rolling_kernel_twap')
        res = update_rolling_kernel_twap(timestamps,
                                         dscode,
                                         values,
                                         units,
                                         self.position,
                                         self.values,
                                         self.units,
                                         self.last_ts,
                                         self.window,
                                         self.spike_interval,
                                         self.half_life,
                                         self.spike_width,
                                         )
        self.save()
        print('finished running update_rolling_kernel_twap')
        return res


if __name__ == '__main__':
    import pandas as pd

    l = []
    for t in range(300):
        # t, spike_interval_ms, half_life_ms, spike_width_ms
        p = create_continuous_smooth_kernel(t, 20, 30, 3)
        l += [p]
    pd.Series(l).to_csv('~/Downloads/kernel.csv')
