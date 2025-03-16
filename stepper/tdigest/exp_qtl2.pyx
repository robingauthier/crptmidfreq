cimport cython
cimport numpy as np
import numpy as np
from .tdigestloc import TDigest


def expanding_quantile( dt, dscode, serie, qs,tdigest_map):
    """
    Update the T-Digest state with new data in (datetime, code, value) arrays.
    Return a 2D numpy array (n_rows x len(qs)) of quantile values for each row.
    
    Parameters
    ----------
    dt : array_like
        Timestamps (unused here, kept for API consistency)
    dscode : array_like
        An array or list of code values (e.g. strings or ints)
    serie : array_like
        An array of float values.
    qs : list
        List of quantiles (as fractions, e.g. [0.1, 0.5, 0.9])
    
    Returns
    -------
    np.ndarray
        A 2D array with shape (n, len(qs)) containing the quantile estimates.
    """
    cdef int compression = 1000
    cdef int n = len(serie)
    cdef int n_qs = len(qs)
    cdef np.ndarray[np.double_t, ndim=2] results = np.zeros((n, n_qs), dtype=np.float64)
    cdef int i, j
    cdef double val, q_percent
    cdef object code
    cdef np.ndarray q_arr
    cdef np.ndarray out_arr


    for i in range(n):
        code = dscode[i]
        val = serie[i]
        if code not in tdigest_map:
            tdigest_map[code] = TDigest()

        # Update the digest with the new value (weight 1.0)
        tdigest_map[code].add(val,1.0)

        for j in range(n_qs):
            q_percent = qs[j] #* 100.0  # Convert quantile fraction to percentage
            results[i, j] = tdigest_map[code].quantile(q_percent)
    return results
