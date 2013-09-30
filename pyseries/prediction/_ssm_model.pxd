# -*- coding: utf8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from __future__ import division, print_function

from cpython cimport bool

import numpy as np
cimport numpy as np

cdef class Model(object):
    
    cdef double min_err
    cdef double alpha
    cdef double beta
    cdef double d
    cdef double gamma
    cdef double std
    cdef int m

    cdef double[:] y
    cdef double[:] b
    cdef double[:] l
    cdef double[:] s
