from libc.string cimport memcpy
from numpy.math cimport NAN, isfinite
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from cpython.dict cimport (PyDict_Check, PyDict_GetItem, PyDict_Merge,
                           PyDict_New, PyDict_Next, PyDict_SetItem,
                           PyDict_Update)
from copy import copy

cimport cython
cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "tdigest_stubs.c":
    ctypedef struct centroid_t:
        double mean
        double weight

    # Not the full struct, just the parameters we want
    ctypedef struct tdigest_t:
        double compression
        size_t size
        size_t buffer_size

        double min
        double max

        size_t ncentroids
        double total_weight
        double buffer_total_weight
        centroid_t *centroids

    tdigest_t *tdigest_new(double compression)
    void tdigest_free(tdigest_t *T)
    void tdigest_add(tdigest_t *T, double x, double w)
    void tdigest_flush(tdigest_t *T)
    void tdigest_merge(tdigest_t *T, tdigest_t *other)
    void tdigest_scale(tdigest_t *T, double factor)
    np.npy_intp tdigest_update_ndarray(tdigest_t *T, np.PyArrayObject *x, np.PyArrayObject *w) except -1
    np.PyArrayObject *tdigest_quantile_ndarray(tdigest_t *T, np.PyArrayObject *q) except NULL
    np.PyArrayObject *tdigest_cdf_ndarray(tdigest_t *T, np.PyArrayObject *x) except NULL


CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])


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
    cdef np.PyArrayObject *q_arr_ptr
    cdef np.ndarray out_arr
    cdef tdigest_t *td_ptr


    for i in range(n):
        code = dscode[i]
        val = serie[i]
        if code not in tdigest_map:
            td_ptr = tdigest_new(compression)
            tdigest_map[code] = PyCapsule_New(td_ptr, "tdigest_t", NULL)
        else:
            td_ptr = <tdigest_t*> PyCapsule_GetPointer(tdigest_map[code], "tdigest_t")
        # Update the digest with the new value (weight 1.0)
        tdigest_add(td_ptr, val, 1.0)
        for j in range(n_qs):
            q_percent = qs[j] #* 100.0  # Convert quantile fraction to percentage
            # Create a 1-element numpy array holding q_percent.
            q_arr = np.array([q_percent], dtype=np.float64)
            q_arr_ptr = <np.PyArrayObject*> q_arr
            # Call the C function to compute the quantile.
            out_arr = <np.ndarray> tdigest_quantile_ndarray(td_ptr, q_arr_ptr)
            # out_arr is expected to be a 1-element array; store its first element.
            results[i, j] = out_arr[0]
    return results