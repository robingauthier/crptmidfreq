# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport exp, log

# We assume that BaseStepper is defined somewhere.
# For example, if BaseStepper is a Python class, you can import it as follows:
from crptmidfreq.stepper.base_stepper import BaseStepper

ctypedef np.int64_t int64_t
ctypedef np.float64_t float64_t

##########################################
# 1. Get alpha (convert half-life to alpha)
##########################################

cdef inline double get_alpha(int window):
    """
    Convert half-life to alpha.
    """
    return 1.0 - exp(log(0.5) / window)

#######################################################
# 2. Update EWM kurtosis values (Cython implementation)
#######################################################

def update_ewmkurt_values(np.ndarray[int64_t, ndim=1] codes,
                          np.ndarray[float64_t, ndim=1] values,
                          np.ndarray[int64_t, ndim=1] timestamps,
                          double alpha,
                          dict ewm_values,
                          dict ewm_squared_values,
                          dict ewm_cubed_values,
                          dict ewm_fourth_values,
                          dict last_timestamps):
    """
    Vectorized update of EWM values for kurtosis calculation.

    Parameters:
      codes: 1D NumPy array of int64 codes.
      values: 1D NumPy array of float64 values.
      timestamps: 1D NumPy array of int64 timestamps.
      alpha: smoothing parameter.
      ewm_values, ewm_squared_values, ewm_cubed_values, ewm_fourth_values:
             Python dicts mapping int code -> float (the current EWM state).
      last_timestamps: Python dict mapping int code -> int64 (last timestamp seen).

    Returns:
      1D NumPy array (float64) of updated EWM kurtosis values.
    """
    cdef Py_ssize_t n = codes.shape[0]
    cdef np.ndarray[float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef np.ndarray[int64_t, ndim=1] count = np.zeros(n, dtype=np.int64)
    cdef dict code_counts = {}
    cdef Py_ssize_t i
    cdef int code, ct, last_ts
    cdef double value, squared_value, cubed_value, fourth_value
    cdef double old_value, old_squared_value, old_cubed_value, old_fourth_value
    cdef double new_value, new_squared_value, new_cubed_value, new_fourth_value
    cdef double variance, m4, kurt
    cdef int64_t ts

    # First pass: count occurrences of each code up to each index
    for i in range(n):
        code = <int> codes[i]
        if code in code_counts:
            ct = code_counts[code]
        else:
            ct = 0
        ct += 1
        count[i] = ct
        code_counts[code] = ct

    # Main loop: update EWM values and compute kurtosis
    for i in range(n):
        code = <int> codes[i]
        value = values[i]
        squared_value = value * value
        cubed_value = squared_value * value
        fourth_value = squared_value * squared_value
        ts = timestamps[i]

        # Ensure timestamps increase per code
        if code in last_timestamps:
            last_ts = last_timestamps[code]
        else:
            last_ts = 0
        if ts < last_ts:
            raise ValueError("DateTime must be strictly increasing per code")

        # Get previous EWM values; if not present, set to NaN
        if code in ewm_values:
            old_value = ewm_values[code]
            old_squared_value = ewm_squared_values[code]
            old_cubed_value = ewm_cubed_values[code]
            old_fourth_value = ewm_fourth_values[code]
        else:
            old_value = float('nan')
            old_squared_value = float('nan')
            old_cubed_value = float('nan')
            old_fourth_value = float('nan')

        if np.isnan(old_value):
            # First occurrence: initialize with the observed values.
            new_value = value
            new_squared_value = squared_value
            new_cubed_value = cubed_value
            new_fourth_value = fourth_value
            kurt = 0.0
        else:
            # Exponential update using decay factor alpha.
            new_value = old_value * (1.0 - alpha) + value * alpha
            new_squared_value = old_squared_value * (1.0 - alpha) + squared_value * alpha
            new_cubed_value = old_cubed_value * (1.0 - alpha) + cubed_value * alpha
            new_fourth_value = old_fourth_value * (1.0 - alpha) + fourth_value * alpha

            variance = new_squared_value - new_value * new_value
            if variance > 1e-14:
                m4 = new_fourth_value - 4.0 * new_cubed_value * new_value + 6.0 * new_squared_value * new_value * new_value - 3.0 * new_value * new_value * new_value * new_value
                kurt = m4 / (variance * variance) - 3.0
            else:
                kurt = 0.0

        # Save updated values
        ewm_values[code] = new_value
        ewm_squared_values[code] = new_squared_value
        ewm_cubed_values[code] = new_cubed_value
        ewm_fourth_values[code] = new_fourth_value
        last_timestamps[code] = ts

        result[i] = kurt

    return result

#######################################################
# 3. Cython version of the EwmKurtStepper class
#######################################################

cdef class EwmKurtStepper(BaseStepper):
    """
    A Cython-based implementation of EWM kurtosis computation.
    """
    cdef public int window
    cdef public double alpha
    cdef public dict ewm_values
    cdef public dict ewm_squared_values
    cdef public dict ewm_cubed_values
    cdef public dict ewm_fourth_values
    cdef public dict last_timestamps

    def __init__(self, folder="", name="", int window=1):
        # Initialize the BaseStepper (assumes BaseStepper.__init__(self, folder, name) exists)
        BaseStepper.__init__(self, folder, name)
        self.window = window
        self.alpha = get_alpha(window)
        # Initialize state dictionaries as empty Python dicts
        self.ewm_values = {}
        self.ewm_squared_values = {}
        self.ewm_cubed_values = {}
        self.ewm_fourth_values = {}
        self.last_timestamps = {}

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, int window=1):
        """
        Load instance from saved state or create a new one if it does not exist.
        """
        return cls.load_utility(cls, folder=folder, name=name, window=window)

    def update(self, dt, dscode, serie):
        """
        Update EWM kurtosis values for each code.

        Parameters:
          dt: numpy array of datetime64 values.
          dscode: numpy array of categorical (int64) codes.
          serie: numpy array of float64 values to process.

        Returns:
          numpy array of float64 with the EWM kurtosis for each input row.
        """
        self.validate_input(dt, dscode, serie)
        # Convert dt (datetime64) to its int64 representation
        cdef np.ndarray[int64_t, ndim=1] ts_array = dt.view(np.int64)
        return update_ewmkurt_values(dscode, serie, ts_array, self.alpha,
                                     self.ewm_values, self.ewm_squared_values,
                                     self.ewm_cubed_values, self.ewm_fourth_values,
                                     self.last_timestamps)
