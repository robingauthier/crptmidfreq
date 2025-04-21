# p2_expanding_quantile.pyx
cimport cython
cimport numpy as np

import numpy as np

# Import our Cython wrapper (adjust the import path as needed)

from crptmidfreq.stepper.p2_algo.p2_quantile cimport P2Quantile


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef expanding_quantile(
    np.ndarray dt, 
    np.ndarray dscode, 
    np.ndarray[double, ndim=1] serie,
    np.ndarray[double, ndim=1] qs, 
    dict p2_map, 
    int freq, 
    dict last_values, 
    dict last_i,
    np.ndarray[np.double_t, ndim=2] results,
    ):
    """
    Update the P² state with new data in (datetime, code, value) arrays.
    Return a 2D numpy array (n_rows x len(qs)) of quantile estimates for each row.
    
    Parameters
    ----------
    dt : array_like
        Timestamps (unused, kept for API consistency)
    dscode : numpy.ndarray
        Array of code values (will be converted to int64)
    serie : numpy.ndarray[double]
        Array of float values.
    qs : numpy.ndarray[double]
        Array of quantiles (fractions, e.g. [0.1, 0.5, 0.9])
    p2_map : dict
        Dictionary mapping code -> P2Quantile instance.
    freq : int
        Frequency (block size) at which to update the P² structure.
    last_values : dict
        Dictionary mapping code -> 1D numpy array of buffered values.
    last_i : dict
        Dictionary mapping code -> last index in the buffer.
    
    Returns
    -------
    np.ndarray
        2D array with shape (n, len(qs)) containing quantile estimates.
    """
    cdef int n = serie.shape[0]
    cdef int n_qs = qs.shape[0]
    # Allocate output array.
    #cdef np.ndarray[np.double_t, ndim=2] results = np.zeros((n, n_qs), dtype=np.float64)
    cdef int i, j, iloc
    cdef double q_percent, val
    # We'll assume dscode can be converted to an int64 memoryview.
    cdef np.ndarray[np.int64_t, ndim=1] codes = dscode.astype(np.int64)
    cdef int code
    cdef P2Quantile p2q

    for i in range(n):
        code = codes[i]
        val = serie[i]
        # Initialize state for new code.
        if code not in p2_map:
            p2_map[code] = P2Quantile(qs[0])
            last_values[code] = np.zeros(freq, dtype=np.float64)
            last_i[code] = 0

        # Increment and update the buffered values.
        last_i[code] = last_i[code] + 1
        iloc = last_i[code]
        last_values[code][iloc % freq] = val

        if iloc >= freq:
            p2q = <P2Quantile>p2_map[code]
            # Use the buffered block as a contiguous array.
            print(last_values[code])
            p2q.update(last_values[code].view(np.double))
            for j in range(n_qs):
                q_percent = qs[j]*100.0
                results[i, j] = p2q.quantile_with(q_percent)
            last_i[code] = 0  # Reset the buffer for this code.
            last_values[code] = np.zeros(freq, dtype=np.float64)
        else:
            # If not enough new samples, forward-fill from the previous row.
            for j in range(n_qs):
                results[i, j] = np.nan
    return results
    