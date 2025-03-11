import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from config_loc import get_data_folder

# Wavelet filter coefficients
WAVELETS = {
    'haar': {
        'dec_lo': np.array([0.7071067811865476, 0.7071067811865476]),
        'dec_hi': np.array([-0.7071067811865476, 0.7071067811865476]),
        'rec_lo': np.array([0.7071067811865476, 0.7071067811865476]),
        'rec_hi': np.array([0.7071067811865476, -0.7071067811865476])
    },
    'db4': {
        'dec_lo': np.array([-0.010597401784997278, 0.032883011666982945,
                            0.030841381835986965, -0.187034811718881490,
                            -0.027983769416983849, 0.630880767929859800,
                            0.714846570552541500, 0.230377813308855400]),
        'dec_hi': np.array([-0.230377813308855400, 0.714846570552541500,
                            -0.630880767929859800, -0.027983769416983849,
                            0.187034811718881490, 0.030841381835986965,
                            -0.032883011666982945, -0.010597401784997278]),
        'rec_lo': np.array([0.230377813308855400, 0.714846570552541500,
                            0.630880767929859800, -0.027983769416983849,
                            -0.187034811718881490, 0.030841381835986965,
                            0.032883011666982945, -0.010597401784997278]),
        'rec_hi': np.array([-0.010597401784997278, -0.032883011666982945,
                            0.030841381835986965, 0.187034811718881490,
                            -0.027983769416983849, -0.630880767929859800,
                            0.714846570552541500, -0.230377813308855400])
    }
}


@njit
def pad_signal(x, filter_len):
    """Pad signal for wavelet transform"""
    pad_len = filter_len - 1
    return np.concatenate([x[-pad_len:], x, x[:pad_len]])


@njit
def conv_down(x, h):
    """Convolution followed by downsampling"""
    N = len(x)
    h_len = len(h)
    y = np.zeros((N + h_len - 1) // 2)

    for n in range(0, N - h_len + 1, 2):
        y[n // 2] = np.sum(x[n:n + h_len] * h)
    return y


@njit
def conv_up(x, h, n):
    """Upsampling followed by convolution"""
    x_up = np.zeros(n)
    x_up[::2] = x
    y = np.zeros(n)
    h_len = len(h)

    for i in range(n):
        k_min = max(0, i - h_len + 1)
        k_max = min(i + 1, n)
        for k in range(k_min, k_max):
            if i - k < h_len:
                y[i] += x_up[k] * h[i - k]
    return y


@njit
def dwt_forward(data, dec_lo, dec_hi, level):
    """
    Forward discrete wavelet transform
    Returns: approximation coefficients, list of detail coefficients
    """
    coeffs = []
    a = data.copy()

    for _ in range(level):
        if len(a) < len(dec_lo):
            break

        # Pad signal
        a_pad = pad_signal(a, len(dec_lo))

        # Decomposition
        a_next = conv_down(a_pad, dec_lo)
        d = conv_down(a_pad, dec_hi)

        coeffs.append(d)
        a = a_next

    coeffs.append(a)
    return coeffs


@njit
def dwt_inverse(coeffs, rec_lo, rec_hi):
    """
    Inverse discrete wavelet transform
    """
    a = coeffs[-1]
    for d in coeffs[-2::-1]:
        n = (len(a) + len(d)) * 2
        a_up = conv_up(a, rec_lo, n)
        d_up = conv_up(d, rec_hi, n)
        a = a_up + d_up
    return a


class WaveletResidualStepper:
    def __init__(self, folder='', name='', window_size=64, wavelet_type='db4', level=3):
        """
        Initialize WaveletResidualStepper

        Args:
            folder: folder to save/load state
            name: name of this instance
            window_size: size of wavelet window (must be power of 2)
            wavelet_type: type of wavelet ('haar' or 'db4')
            level: number of decomposition levels
        """
        if not (window_size & (window_size - 1) == 0):
            raise ValueError("Window size must be a power of 2")

        self.folder = os.path.join(get_data_folder(), folder)
        self.name = name
        self.window_size = window_size
        self.wavelet_type = wavelet_type
        self.level = level

        # Get wavelet filters
        wavelet = WAVELETS[wavelet_type]
        self.dec_lo = wavelet['dec_lo']
        self.dec_hi = wavelet['dec_hi']
        self.rec_lo = wavelet['rec_lo']
        self.rec_hi = wavelet['rec_hi']

        # Initialize empty state
        self.last_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],
        )
        self.last_coeffs = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:],
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )

    def process_signal(self, signal):
        """Process single signal through wavelet transform"""
        # Forward transform
        coeffs = dwt_forward(signal, self.dec_lo, self.dec_hi, self.level)

        # Modify coefficients (e.g., threshold small details)
        modified_coeffs = coeffs.copy()
        for i in range(len(modified_coeffs) - 1):  # Don't modify approximation
            modified_coeffs[i] = np.where(
                np.abs(modified_coeffs[i]) < np.std(modified_coeffs[i]),
                0,
                modified_coeffs[i]
            )

        # Inverse transform
        reconstructed = dwt_inverse(modified_coeffs, self.rec_lo, self.rec_hi)

        # Calculate residual
        residual = signal - reconstructed
        return residual, coeffs

    def update(self, dt, dscode, serie):
        """
        Update wavelet residuals for each code
        """
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have same length")

        if not dt.dtype == 'int64':
            timestamps = dt.astype('datetime64[ns]').astype('int64')
        else:
            timestamps = dt

        result = np.empty(len(dscode), dtype=np.float64)

        for i in range(len(dscode)):
            code = dscode[i]
            value = serie[i]
            ts = timestamps[i]

            # Check timestamp ordering
            last_ts = self.last_timestamps.get(code, np.int64(0))
            if ts < last_ts:
                raise ValueError("DateTime must be strictly increasing per code")

            if np.isnan(value):
                result[i] = np.nan
                self.last_timestamps[code] = ts
                continue

            # Get or initialize buffer
            current_values = self.last_values.get(code, np.zeros(self.window_size))

            # Update buffer
            current_values = np.roll(current_values, -1)
            current_values[-1] = value

            # Process through wavelet transform
            residual, coeffs = self.process_signal(current_values)

            # Store state
            self.last_values[code] = current_values
            self.last_coeffs[code] = coeffs[-1]  # Store approximation coefficients
            self.last_timestamps[code] = ts

            # Store result
            result[i] = residual[-1]  # Return residual for newest value

        self.save()
        return result

    # Save and load methods remain the same

    def save(self):
        """Save internal state to file"""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        state = {
            'last_timestamps': dict(self.last_timestamps),
            'last_values': {k: v.copy() for k, v in self.last_values.items()},
            'last_coeffs': {k: v.copy() for k, v in self.last_coeffs.items()},
            'window_size': self.window_size,
        }

        filepath = os.path.join(self.folder, self.name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, folder, name, window_size=None):
        """Load instance from saved state or create new if not exists"""
        instance_key = f"{folder}/{name}"

        folder_path = os.path.join(get_data_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            instance = cls(folder=folder, name=name, window_size=window_size)
            return instance

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create new instance
        instance = cls(folder=folder_path, name=name, window_size=state['window_size'])

        # Convert regular dicts back to numba Dicts
        for k, v in state['last_values'].items():
            instance.last_values[k] = v
        for k, v in state['last_coeffs'].items():
            instance.last_coeffs[k] = v
        for k, v in state['last_timestamps'].items():
            instance.last_timestamps[k] = v

        cls._instances[instance_key] = instance
        return instance
