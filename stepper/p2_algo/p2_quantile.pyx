# p2_quantile.pyx
# distutils: language = c++

cimport cython
cimport numpy as np
import numpy as np


# Wrap the C++ class in a Cython class.
cdef class P2Quantile:
    """
    A Cython wrapper for the P² algorithm.
    """

    def __cinit__(self, double target):
        self._target = target
        self._p2 = new p2_t(target)

    def __dealloc__(self):
        if self._p2 is not NULL:
            del self._p2

    cdef void add(self, double data):
        self._p2.add(data)

    cdef void update(self, double[:] data):
        """
        Update the P2 structure with a block of data. Note that the weights array is ignored 
        (the basic P² algorithm does not support weights).
        """
        cdef int n = data.shape[0]
        for i in range(n):
            self.add(data[i])

    cdef double quantile_with(self, double q):
        """
        Return the quantile value corresponding to q (as a fraction).
        """
        return self._p2.result(q)

    cdef double quantile(self):
        """
        Return the quantile value (using the primary target).
        """
        return self._p2.result()

    def __getstate__(self):
        """
        Minimal pickling support. This only saves the target quantile and the current estimate.
        (It does not capture the full internal state.)
        """
        return (self._target, self.quantile())

    def __setstate__(self, state):
        self.__cinit__(state[0])
        # Optionally, you might want to re-add data to approximate the previous state.
