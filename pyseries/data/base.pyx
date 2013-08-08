# -*- coding: utf8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division, print_function
'''
Base objects for working with time series datasets. Here we keep the objects
used to manipulated time series datasets:
    * TimeSeries : Class for a single time series
    * TimeSeriesDataset : Dataset for of various time series
'''

from pyseries.exceptions import DataFormatException
from pyseries.exceptions import ParameterException

import numpy as np
cimport numpy as np

cdef inline int TRUE = 1
cdef inline int FALSE = 0

cdef inline Py_ssize_t smax(Py_ssize_t first, Py_ssize_t second):
    return first if first >= second else second

cdef inline Py_ssize_t smin(Py_ssize_t first, Py_ssize_t second):
    return first if first < second else second

cdef int check_unique_sorted(double[:] array) nogil:
    '''Checks if the given array is sorted'''

    cdef Py_ssize_t i
    for i from 1 <= i < array.shape[0]:
        if array[i] <= array[i - 1]:
            return FALSE

    return TRUE

cdef Py_ssize_t bin_search_pos(double[:] array, double value) nogil:
    '''
    Finds the first element in the array where the given is OR should have been
    in the given array. This is simply a binary search, but if the element is
    not found we return the index where it should have been at.

    This method is mainly used to filter dates from a time series.
    '''

    cdef Py_ssize_t n = array.shape[0]
    cdef Py_ssize_t lower = 0
    cdef Py_ssize_t upper = n - 1 #closed interval
    
    cdef Py_ssize_t half = 0
    cdef Py_ssize_t idx = -1 
    while upper >= lower:
        half = lower + ((upper - lower) // 2)
        if value == array[half]:
            idx = half
            break
        elif value > array[half]:
            lower = half + 1
        else:
            upper = half - 1
    
    if idx == -1: #Element not found, return where it should be
        idx = lower - 1

    return idx

cdef class TimeSeries(object):
    '''
    Represents a single time series

    Parameters
    ----------
    data : array like of double
        data of the time series
    timestamps : array like of double
        time stamp (in seconds since epoch) for each event. This array must
        be sorted and unique. An exception is thrown if either conditions are
        not met.
    '''

    def __init__(self, double[:] data, double[:] timestamps):
        if check_unique_sorted(timestamps) == FALSE:
            raise DataFormatException('Timestamps must be sorted and unique')

        if data.shape[0] != timestamps.shape[0]:
            raise ParameterException('Arrays must have the same shape')

        self.data = data
        self.timestamps = timestamps
        self.size = data.shape[0]

    cpdef TimeSeries filter_upper(self, double timestamp):
        '''
        Creates a new TimeSeries object with the elements which exist from 
        (open interval) a given date.
        
        Parameters
        ----------
        timestamp : double
            Date to filter
        '''
        cdef Py_ssize_t idx = bin_search_pos(self.timestamps, timestamp)
        return TimeSeries(self.data[idx:], self.timestamps[idx:])

    cpdef TimeSeries filter_lower(self, double timestamp):
        '''
        Creates a new TimeSeries object with the elements which exist up to
        (open interval) a given date.
        
        Parameters
        ----------
        timestamp : double
            Date to filter
        '''
        cdef Py_ssize_t idx = bin_search_pos(self.timestamps, timestamp)
        return TimeSeries(self.data[:idx], self.timestamps[:idx])

    cpdef TimeSeries filter_mid(self, double lowerstamp, double upperstamp):
        '''
        Creates a new TimeSeries object with elements which exist from (closed)
        a given up to (open) a given date.

        Parameters
        ----------
        lowerstamp : double
            Lower date
        upperstamp : double
            Upper date
        '''
        cdef Py_ssize_t lower = bin_search_pos(self.data, lowerstamp)
        cdef Py_ssize_t upper = bin_search_pos(self.data, upperstamp)
        return TimeSeries(self.data[lower:upper], self.timestamps[lower:upper])

    cdef Py_ssize_t size(self):
        '''Get's the number of elements in this time series'''
        return self.size

    def __len__(self):
        return self.size

cdef class TimeSeriesDataset(object):
    '''
    Represents a dataset with multiple time series. Each individual time series
    must be added to the dataset.
    '''
    def __init__(self, TimeSeries[:] series):
        self.series = series
        self.num_series = series.shape[0]
        self.max_size = 0
        self.min_size = 0

        cdef Py_ssize_t i
        for i from 0 <= i < series.shape[0]:
            self.max_size = smax(self.max_size, series.size())
            self.min_size = smin(self.min_size, series.size())

    def __getitem__(self, Py_ssize_t idx):
        return self.series[idx]

    cpdef cnp.ndarray[double, ndim=2] np_like_firstn(self):
        '''
        Converts the time series dataset to a numpy array of
        2 dimensions. Since time series may have different shapes,
        only the first n elements are used, where n is the size of the
        smallest series.
        '''
        cdef cnp.ndarray[double, ndim=2] return_val = \
                cnp.ndarray(shape=(self.num_series, self.min_size))
        
        cdef TimeSeries time_series
        cdef Py_ssize_t i

        for i from 0 <= i < self.num_series:
            time_series = self.series[i]
            return_val[i] = time_series.data[:self.min_size]

        return return_val

    cpdef cnp.ndarray[double, ndim=2] np_like_lastn(self):
        '''
        Converts the time series dataset to a numpy array of
        2 dimensions. Since time series may have different shapes,
        only the last n elements are used, where n is the size of the
        largest series.
        '''

        cdef cnp.ndarray[double, ndim=2] return_val = \
                cnp.ndarray(shape=(self.num_series, self.min_size))
        
        cdef Py_ssize_t start_idx
        cdef TimeSeries time_series
        cdef Py_ssize_t i

        for i from 0 <= i < self.num_series:
            time_series = self.series[i]
            start_idx = time_series.data.shape[0] - self.min_size
            return_val[i] = time_series.data[start_idx:]

        return return_val
