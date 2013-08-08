# -*- coding: utf8
from __future__ import division, print_function

cimport numpy as cnp

cdef class TimeSeries(object):
    
    cdef readonly double[:] data
    cdef readonly double[:] timestamps
    cdef readonly Py_ssize_t size

    cpdef TimeSeries filter_lower(self, double timestamp)
    cpdef TimeSeries filter_upper(self, double timestamp)
    cpdef TimeSeries filter_mid(self, double lowerstamp, double upperstamp)
    cdef Py_ssize_t size(self)

cdef class TimeSeriesDataset(object):
    
    cpdef readonly TimeSeries[:] series
    cpdef readonly Py_ssize_t num_series
    cpdef readonly Py_ssize_t min_size
    cpdef readonly Py_ssize_t max_size

    cpdef cnp.ndarray[double, ndim=2] np_like_firstn(self)
    cpdef cnp.ndarray[double, ndim=2] np_like_lastn(self)
