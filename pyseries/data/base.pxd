# -*- coding: utf8
from __future__ import division, print_function

cdef class TimeSeries(object):
    
    cdef double[:] data
    cdef double[:] timestamps
    cdef Py_ssize_t size

    cdef TimeSeries filter_lower(self, double timestamp)
    cdef TimeSeries filter_upper(self, double timestamp)
    cdef TimeSeries filter_mid(self, double lowerstamp, double upperstamp)
    cdef Py_ssize_t size(self)

cdef class TimeSeriesDataset(object):
    
    cdef TimeSeries[:] series
    cdef Py_ssize_t num_series
    cdef Py_ssize_t min_size
    cdef Py_ssize_t max_size

    cdef double[::1] np_like_firstn(self)
    cdef double[::1] np_like_lastn(self)
    cdef double[::1] np_like_round_peak(self, Py_ssize_t peak_round)
