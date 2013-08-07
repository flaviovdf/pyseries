# -*- coding: utf8
from __future__ import division, print_function

cdef class TimeSeries(object):
    
    cdef double[:] data
    cdef double[:] timestamps

    cdef TimeSeries filter_lower(self, double timestamp)
    cdef TimeSeries filter_upper(self, double timestamp)
    cdef TimeSeries filter_mid(self, double lowerstamp, double upperstamp)

cdef class TimeSeriesDataset(object):
    
    cdef TimeSeries[:] series
    cdef Py_ssize_t num_series
    cdef Py_ssize_t min_size
    cdef Py_ssize_t max_size

    cdef double[::1] to_numpy_like_data(self, str heuristic)
