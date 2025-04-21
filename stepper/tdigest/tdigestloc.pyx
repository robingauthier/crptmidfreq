
# https://github.com/dask/crick/blob/main/crick/tdigest.pyx
from libc.string cimport memcpy
from numpy.math cimport NAN, isfinite

from copy import copy

cimport cython
cimport numpy as np

import numpy as np

np.import_array()


CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])


@cython.boundscheck(False)
cdef inline void _cdf_to_hist(double[:] cdf, double[:]  hist, double size, int hist_size):
    cdef size_t i
    for i in range(hist_size):
        hist[i] = (cdf[i + 1] - cdf[i]) * size




# cython: boundscheck=False, wraparound=False, nonecheck=False

cimport cython
# Import only what we need from the C standard library
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy


# Extern declarations for the underlying C T‑Digest implementation
cdef extern from "tdigest_stubs.c":
    ctypedef struct centroid_t:
        double mean
        double weight

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
    # Note: Other functions (e.g. for quantile, cdf, update_ndarray) are omitted
    # in order to keep everything pure C and fast. For bulk updates we will loop in C.


    
cdef class TDigest:
    """
    A pure-C (Cython) wrapper for a T-Digest.
    
    This version removes Python-level overhead: it assumes that all inputs
    (for bulk update) are passed as contiguous memoryviews of type double.
    """

    cdef void add(self, double x, double w):
        """
        Add a single sample (x) with weight (w) to the digest.
        Pure C call with no Python overhead.
        """
        # Caller must ensure w > 0 (no runtime check here)
        tdigest_add(self.tdig, x, w)

    cdef void update(self, double[:] x):
        """
        Bulk-update: add many samples in one call.
        Both x and w must be contiguous C-level memoryviews of type double
        and of equal length.
        """
        cdef Py_ssize_t i, n = x.shape[0]
        cdef double w =1.0
        # No error checking here: assume inputs are correct.
        for i in range(n):
            tdigest_add(self.tdig, x[i], w)


    cdef double quantile(self, double q):
        """quantile(self, q)

        Compute an estimate of the qth percentile of the data added to
        this digest.

        Parameters
        ----------
        q : array_like or float
            A number between 0 and 1 inclusive.
        """
        cdef np.ndarray arr_q = np.asarray(q)
        out = <np.ndarray>tdigest_quantile_ndarray(self.tdig, <np.PyArrayObject*>arr_q)
        if out.ndim == 0:
            return np.float64(out)
        return out

    cdef double get_min(self):
        """
        Get the minimum value. Flushes internal buffers.
        """
        tdigest_flush(self.tdig)
        return self.tdig.min

    cdef double get_max(self):
        """
        Get the maximum value. Flushes internal buffers.
        """
        tdigest_flush(self.tdig)
        return self.tdig.max

    cdef double total_weight(self):
        """
        Return the total weight in the digest.
        """
        return self.tdig.total_weight + self.tdig.buffer_total_weight

    cdef size_t ncentroids(self):
        """
        Return the number of centroids (after flushing).
        """
        tdigest_flush(self.tdig)
        return self.tdig.ncentroids

    # Optionally, if you wish to expose centroids as a C-level block of memory,
    # you could add a method to copy them into a caller-supplied buffer.
    cdef void copy_centroids(self, centroid_t *dest, size_t n):
        """
        Copy up to n centroids into dest.
        Caller is responsible for ensuring that dest has enough space.
        """
        cdef size_t m
        tdigest_flush(self.tdig)
        m = self.tdig.ncentroids
        if m > n:
            m = n
        memcpy(dest, self.tdig.centroids, m * sizeof(centroid_t))

    @property
    def compression(self):
        """The compression factor for this digest"""
        return self.tdig.compression


    def __cinit__(self, double compression=100.0):
        # (Minimal checking: caller must ensure compression > 0)
        if compression <= 0:
            raise ValueError("compression must be > 0")
        self.tdig = tdigest_new(compression)
        if self.tdig == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.tdig != NULL:
            tdigest_free(self.tdig)


    def __getstate__(self):
        return (self.centroids(), self.tdig.total_weight,
                self.tdig.min, self.tdig.max)

    def __setstate__(self, state):
        self.tdig.total_weight = <double>state[1]
        self.tdig.min = <double>state[2]
        self.tdig.max = <double>state[3]

        cdef np.ndarray centroids = state[0]
        cdef int n = len(centroids)
        if n > 0:
            memcpy(self.tdig.centroids, centroids.data,
                   n * sizeof(centroid_t))
            self.tdig.ncentroids = n

    def __repr__(self):
        return ("TDigest<compression={0}, "
                "size={1}>").format(self.compression, self.size())


    def __reduce__(self):
        # pickle method !
        #return (TDigest, (self.compression,), self.__getstate__())
        from crptmidfreq.stepper.tdigest.pickle_helper import \
            _reconstruct_TDigest
        return (_reconstruct_TDigest, (self.compression,), self.__getstate__())


    def centroids(self):
        """centroids(self)

        Returns a numpy array of all the centroids in the digest. Note that
        this array is a *copy* of the internal data.
        """
        cdef size_t n
        tdigest_flush(self.tdig)
        n = self.tdig.ncentroids

        cdef np.ndarray[centroid_t, ndim=1] result = np.empty(n, dtype=CENTROID_DTYPE)
        if n > 0:
            memcpy(result.data, self.tdig.centroids, n * sizeof(centroid_t))
        return result

    def size(self):
        """size(self)

        The sum of the weights on all centroids."""
        return self.tdig.total_weight + self.tdig.buffer_total_weight
