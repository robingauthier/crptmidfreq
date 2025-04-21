
cimport cython
cimport numpy as np

import numpy as np


# Declare the P² C++ class from your header.
cdef extern from "p2.h" namespace "":
    cdef cppclass p2_t:
        p2_t() except +
        p2_t(double quantile) except +
        void add_quantile(double quant)
        void add_equal_spacing(int n)
        void add(double data)
        double result()              # when only one quantile is targeted
        double result(double quantile)  # when querying a specific quantile



cdef class P2Quantile:
    cdef p2_t* _p2
    cdef double _target  # the primary quantile target

    cdef void add(self, double data)
    cdef void update(self, double[:] data)
    cdef double quantile_with(self, double q)
    cdef double quantile(self)
