

def expanding_quantile(dt, dscode, serie, qs, p2_map, freq, last_values, last_i):
    """
    Update the P² state with new data in (datetime, code, value) arrays.
    Return a 2D numpy array (n_rows x len(qs)) of quantile estimates for each row.
    
    Parameters
    ----------
    dt : array_like
        Timestamps (unused here, kept for API consistency)
    dscode : array_like
        An array or list of code values (e.g. ints)
    serie : array_like
        An array of float values.
    qs : list or array_like
        List of quantiles (fractions, e.g. [0.1, 0.5, 0.9])
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
    cdef int n = len(serie)
    cdef int n_qs = len(qs)
    cdef np.ndarray[np.double_t, ndim=2] results = np.zeros((n, n_qs), dtype=np.float64)
    cdef int i, j, iloc
    cdef double q_percent, val
    cdef object code
    cdef P2Quantile p2q

    for i in range(n):
        code = dscode[i]
        val = np.float64(serie[i])
        # Initialize state for new code.
        if code not in p2_map:
            # Create a new P2Quantile instance with the primary target set to qs[0].
            # (If you need multiple quantiles per code, you might store a dict or list instead.)
            p2_map[code] = P2Quantile(qs[0])
            last_values[code] = np.zeros(freq, dtype=np.float64)
            last_i[code] = 0

        last_i[code] = last_i[code] + 1
        iloc = last_i[code]
        last_values[code][iloc % freq] = val
        if iloc >= freq:
            # Update the P² structure with the buffered block of values.
            p2q = <P2Quantile>p2_map[code]
            # The update method adds each new value.
            p2q.update(last_values[code].view(np.float64))
            for j in range(n_qs):
                q_percent = qs[j]
                results[i, j] = p2q.quantile_with(q_percent)
            last_i[code] = 0  # Reset the buffer index for this code.
        else:
            # If not enough new samples, forward-fill the previous row's estimates.
            if i > 0:
                for j in range(n_qs):
                    results[i, j] = results[i-1, j]
            else:
                for j in range(n_qs):
                    results[i, j] = 0.0
    return results
