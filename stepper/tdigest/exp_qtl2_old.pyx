cimport cython
cimport numpy as np
import numpy as np
from crptmidfreq.stepper.tdigest.tdigestloc cimport TDigest # enables to have TDigest as a type



def expanding_quantile( dt, dscode, serie, qs,tdigest_map,freq,last_values,last_i):
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
    cdef double q_percent
    cdef object code
    cdef np.ndarray q_arr
    cdef np.ndarray out_arr
    cdef double val
    cdef double w=1.0
    cdef TDigest digestloc 
    cdef int iloc=0

    for i in range(n):
        code = dscode[i]
        val = np.float64(serie[i])

        if code not in tdigest_map:
            tdigest_map[code] = TDigest()
            last_values[code] = np.zeros(freq,dtype=np.float64)
            last_i[code]=0

        last_i[code]=last_i[code]+1
        iloc = last_i[code]
        last_values[code][iloc%freq]=val

        if iloc>freq:
            # Update the digest with the new value (weight 1.0)
            digestloc=<TDigest>tdigest_map[code]
            digestloc.update(
                last_values[code].view(np.float64),
                np.ones(freq,dtype=np.float64))
            for j in range(n_qs):
                q_percent = qs[j] #* 100.0  # Convert quantile fraction to percentage
                results[i, j] = digestloc.quantile(q_percent)
            last_i[code] = 0
        else:
            # we will forward fill
            for j in range(n_qs):
                results[i, j] = results[i-1, j] 
    return results

