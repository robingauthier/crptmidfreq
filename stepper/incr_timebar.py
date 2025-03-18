import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
import pickle
from crptmidfreq.config_loc import get_data_folder
from collections import deque
from enum import Enum


@njit
def update_bar_values(codes, values, timestamps, period_ns, max_hist,
                       bar_counts,bar_highs, bar_lows, bar_opens, bar_closes,
                      bar_times, bar_ids):
    """
    Update bar values for each code and return number of new bars per code

    Args:
        codes: array of categorical codes
        values: array of values to process
        timestamps: array of timestamps (as int64 nanoseconds)
        period_ns: period size in nanoseconds
        max_hist: maximum number of bars to keep in history
        bar_counts: Dict mapping codes to current bar tick counts
        bar_highs: Dict mapping codes to array of high values
        bar_lows: Dict mapping codes to array of low values
        bar_opens: Dict mapping codes to array of open values
        bar_closes: Dict mapping codes to array of close values
        bar_times: Dict mapping codes to array of bar timestamps
        bar_ids: Dict mapping codes to array of bar IDs
        bar_counts_per_code: Dict mapping codes to number of bars

    Returns:
        Dict mapping codes to number of new bars created
    """
    new_bars_count = Dict.empty(key_type=types.int64, value_type=types.int64)

    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        # Get or initialize bar time
        if code not in bar_times:
            bar_start = ts - (ts % period_ns)
            # Initialize arrays for new code
            bar_times[code] = np.array([bar_start], dtype=np.float64)
            bar_ids[code] = np.array([1], dtype=np.int64)
            bar_opens[code] = np.array([value], dtype=np.float64)
            bar_highs[code] = np.array([value], dtype=np.float64)
            bar_lows[code] = np.array([value], dtype=np.float64)
            bar_closes[code] = np.array([value], dtype=np.float64)
            bar_counts[code] = np.array([1], dtype=np.int64)
            continue

        curr_bar_start = bar_times[code][-1]

        # Check if we need to start a new bar
        if ts >= curr_bar_start + period_ns:
            # Start new bar
            bar_start = ts - (ts % period_ns)

            # Update arrays with new bar
            # Append new bar
            bar_times[code] = np.append(bar_times[code], bar_start)
            bar_ids[code] = np.append(bar_ids[code], bar_ids[code][-1] + 1)
            bar_opens[code] = np.append(bar_opens[code], value)
            bar_highs[code] = np.append(bar_highs[code], value)
            bar_lows[code] = np.append(bar_lows[code], value)
            bar_closes[code] = np.append(bar_closes[code], value)
            bar_counts[code] = np.append(bar_counts[code], [1])
            if len(bar_times[code]) > max_hist:
                bar_times[code]=bar_times[code][-max_hist:]
                bar_ids[code] = bar_ids[code][-max_hist:]
                bar_opens[code] = bar_opens[code][-max_hist:]
                bar_highs[code] = bar_highs[code][-max_hist:]
                bar_lows[code] = bar_lows[code][-max_hist:]
                bar_closes[code] = bar_closes[code][-max_hist:]
                bar_counts[code] = bar_counts[code][-max_hist:]
            new_bars_count[code] = new_bars_count.get(code, 0) + 1
        else:
            # Update current bar
            curr_high = max(bar_highs[code][-1], value)
            curr_low = min(bar_lows[code][-1], value)
            # Update arrays
            bar_highs[code][-1] = curr_high
            bar_lows[code][-1] = curr_low
            bar_closes[code][-1] = value
            bar_counts[code][-1]=bar_counts[code][-1]+1
    return new_bars_count


class TimeBarStepper:
    def __init__(self, folder='', name='', period='1min', max_hist=100):
        """
        Initialize TimeBarStepper

        Args:
            folder: folder for saving/loading state
            name: name for saving/loading state
            period: bar period as string (e.g., '1min', '1s', '1h')
            max_hist: maximum number of historical bars to keep in memory
        """
        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name
        self.period = period
        self.max_hist = max_hist

        # Convert period string to nanoseconds
        self.period_ns = self._period_to_ns(period)

        # Bar arrays per code
        self.bar_times = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],  # Array of timestamps
        )
        self.bar_ids = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],  # Array of IDs
        )
        self.bar_highs = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],  # Array of high values
        )
        self.bar_lows = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],  # Array of low values
        )
        self.bar_opens = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],  # Array of open values
        )
        self.bar_closes = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],  # Array of close values
        )
        self.bar_counts = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:],  # Array of count values
        )
    def _period_to_ns(self, period):
        """Convert period string to nanoseconds"""
        unit = period[-1] if period[-1].isalpha() else period[-3:]
        value = int(''.join(c for c in period if c.isdigit()))

        if unit in ['ns', 'n']:
            return value
        elif unit in ['us', 'u']:
            return value * 1000
        elif unit in ['ms', 'm']:
            return value * 1000000
        elif unit == 's':
            return value * 1000000000
        elif unit == 'T':  # minute
            return value * 60 * 1000000000
        elif unit == 'h':
            return value * 3600 * 1000000000
        else:
            raise ValueError(f"Unsupported period unit: {unit}")

    def save(self):
        """Save internal state to file"""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        state = {
            'bar_highs': dict(self.bar_highs),
            'bar_lows': dict(self.bar_lows),
            'bar_opens': dict(self.bar_opens),
            'bar_closes': dict(self.bar_closes),
            'bar_times': dict(self.bar_times),
            'bar_counts':dict(self.bar_counts),
            'bar_ids': dict(self.bar_ids),
            'period': self.period,
            'max_hist': self.max_hist,
        }
        print(state)
        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name, period='1min', max_hist=100):
        """Load instance from saved state"""
        folder = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder, name + '.pkl')

        if not os.path.exists(filepath):
            instance = cls(folder=folder, name=name, period=period,max_hist=max_hist)
            return instance

        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(state)

        # Create a new instance
        instance = cls(
            folder=folder,
            name=name,
            period=state['period'],
            max_hist=state['max_hist'],
        )

        # Convert regular dicts back to numba Dicts
        for k, v in state['bar_highs'].items():
            instance.bar_highs[k] = v
        for k, v in state['bar_lows'].items():
            instance.bar_lows[k] = v
        for k, v in state['bar_opens'].items():
            instance.bar_opens[k] = v
        for k, v in state['bar_closes'].items():
            instance.bar_closes[k] = v
        for k, v in state['bar_times'].items():
            instance.bar_times[k] = v
        for k, v in state['bar_ids'].items():
            instance.bar_ids[k] = v
        for k, v in state['bar_counts'].items():
            instance.bar_counts[k] = v

        return instance

    def update(self, dt, dscode, serie):
        """
        Update bar values for each code and return the latest bars

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            Tuple of arrays (dt, dscode, id, open, high, low, close) for new bars
        """
        # Input validation
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        # Convert datetime64 to int64 nanoseconds for Numba
        timestamps = dt.astype('datetime64[ns]').astype('int64')

        # Update values and get new bars count using numba function
        new_bars_count = update_bar_values(
            dscode, serie, timestamps, self.period_ns, self.max_hist,
             self.bar_counts,
            self.bar_highs, self.bar_lows, self.bar_opens, self.bar_closes,
            self.bar_times, self.bar_ids
        )

        # Collect new bars
        last_dts = []
        last_dscodes = []
        last_ids = []
        last_opens = []
        last_highs = []
        last_lows = []
        last_closes = []
        last_counts = []

        for code in self.bar_times:
            count = new_bars_count[code]
            if count > 0:
                # Get the last 'count' bars from each array
                last_dts.extend(self.bar_times[code][-count:])
                last_dscodes.extend([code] * count)
                last_ids.extend(self.bar_ids[code][-count:])
                last_opens.extend(self.bar_opens[code][-count:])
                last_highs.extend(self.bar_highs[code][-count:])
                last_lows.extend(self.bar_lows[code][-count:])
                last_closes.extend(self.bar_closes[code][-count:])
                last_counts.extend(self.bar_counts[code][-count:])

        return {'date': np.array(last_dts, dtype=np.int64),
                'dscode': np.array(last_dscodes, dtype=np.int64),
                'id': np.array(last_ids, dtype=np.int64),
                'open': np.array(last_opens, dtype=np.float64),
                'high': np.array(last_highs, dtype=np.float64),
                'cnt': np.array(last_counts, dtype=np.float64),
                'low': np.array(last_lows, dtype=np.float64),
                'close': np.array(last_closes, dtype=np.float64)}

    def get_all_bars(self):
        """
        Get all bars currently in memory for all codes

        Returns:
            Tuple of arrays (dt, dscode, id, open, high, low, close) containing all bars in memory
        """
        # Count total bars
        total_bars = 0
        for code in self.bar_times:
            total_bars += len(self.bar_times[code])

        if total_bars == 0:
            empty_float = np.array([], dtype=np.float64)
            result = empty_float
            return result

        # Pre-allocate arrays
        all_dts = np.empty(total_bars, dtype=np.int64)
        all_dscodes = np.empty(total_bars, dtype=np.int64)
        all_ids = np.empty(total_bars, dtype=np.int64)
        all_opens = np.empty(total_bars, dtype=np.float64)
        all_highs = np.empty(total_bars, dtype=np.float64)
        all_lows = np.empty(total_bars, dtype=np.float64)
        all_closes = np.empty(total_bars, dtype=np.float64)
        all_counts = np.empty(total_bars, dtype=np.float64)

        # Fill arrays
        idx = 0
        for code in self.bar_times:
            count = len(self.bar_times[code])
            all_dts[idx:idx + count] = self.bar_times[code]
            all_dscodes[idx:idx + count] = code
            all_ids[idx:idx + count] = self.bar_ids[code]
            all_opens[idx:idx + count] = self.bar_opens[code]
            all_highs[idx:idx + count] = self.bar_highs[code]
            all_lows[idx:idx + count] = self.bar_lows[code]
            all_closes[idx:idx + count] = self.bar_closes[code]
            all_counts[idx:idx + count] = self.bar_counts[code]
            idx += count
        return {'date': all_dts,
                'dscode': all_dscodes,
                'id': all_ids,
                'open': all_opens,
                'high': all_highs,
                'cnt': all_counts,
                'low': all_lows,
                'close': all_closes}
