import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from config_loc import get_data_folder


@njit
def update_rolling_corr(timestamps,
                        dscode,
                        values1,
                        values2,
                        position,
                        rolling_dict1,
                        rolling_dict2,
                        last_timestamps,
                        window):
    """

    """
    result = np.zeros(len(dscode), dtype=np.float64)
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

        result[i] = np.corrcoef(np.nan_to_num(rolling_dict1[code]),
                                np.nan_to_num(rolling_dict2[code]))[0, 1]
    return result


class RollingCorrStepper:
    def __init__(self, folder='', name='', window=1):
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name
        self.window = window
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
        """Save internal state to file"""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        state = {
            'last_ts': {k: v for k, v in self.last_ts.items()},
            'position': {k: v for k, v in self.position.items()},
            'values1': {k: list(v) for k, v in self.values1.items()},
            'values2': {k: list(v) for k, v in self.values2.items()},
            'window': self.window
        }

        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name='', window=None):
        """Load instance from saved state or create new if not exists"""
        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            print(f'RollingStepper creating instance {folder} {name} {window}')
            return cls(folder=folder, name=name, window=window)

        print(f'RollingStepper loading instance {folder} {name}')
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        instance = cls(folder=folder_path, name=name, window=state['window'])
        instance.values1 = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        instance.values2 = Dict.empty(
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
        for k, v in state['values1'].items():
            instance.values1[k] = np.array(v)
        for k, v in state['values2'].items():
            instance.values2[k] = np.array(v)
        for k, v in state['last_ts'].items():
            instance.last_ts[k] = v
        for k, v in state['position'].items():
            instance.position[k] = v
        return instance

    def update(self, dt, dscode, values1, values2):
        if len(dscode) != len(values1):
            raise ValueError("Codes and values arrays must have the same length")
        if len(dt) != len(values1):
            raise ValueError("Codes and values arrays must have the same length")
        if len(values2) != len(values1):
            raise ValueError("Codes and values arrays must have the same length")

        if not dt.dtype == 'int64':
            timestamps = dt.astype('datetime64[ns]').astype('int64')
        else:
            timestamps = dt

        res = update_rolling_corr(timestamps, dscode, values1, values2,
                                  self.position, self.values1, self.values2,
                                  self.last_ts, self.window)
        self.save()
        return res
