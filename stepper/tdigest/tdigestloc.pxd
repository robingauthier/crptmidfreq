# tdigestloc.pxd

# C extern declarations from "tdigest_stubs.c"
cimport numpy as np
#from libc.stdlib cimport size_t
from libc.stddef cimport size_t


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


# Declaration of the TDigest class.
cdef class TDigest:
    """
    A pure-C (Cython) wrapper for a T-Digest.
    """
    cdef tdigest_t *tdig

    cdef void add(self, double x, double w)
    cdef void update(self, double[:] x)
    cdef double quantile(self, double q)
    cdef double get_min(self)
    cdef double get_max(self)
    cdef double total_weight(self)
    cdef size_t ncentroids(self)
    cdef void copy_centroids(self, centroid_t *dest, size_t n)

