# -*- coding: utf8
from __future__ import division, print_function

import numpy as np
cimport numpy as np
np.import_array()

cdef class TimeSeries(object):
    
    cdef double[:] _timestamps
    cdef double[:] _data
    cdef Py_ssize_t _size

    cpdef TimeSeries filter_lower(self, double timestamp)
    cpdef TimeSeries filter_upper(self, double timestamp)
    cpdef TimeSeries filter_mid(self, double lowerstamp, double upperstamp)

cdef class TimeSeriesDataset(object):
    
    cpdef readonly TimeSeries[:] series
    cpdef readonly Py_ssize_t num_series
    cpdef readonly Py_ssize_t min_size
    cpdef readonly Py_ssize_t max_size

    cpdef np.ndarray[double, ndim=2] np_like_firstn(self)
    cpdef np.ndarray[double, ndim=2] np_like_lastn(self)
