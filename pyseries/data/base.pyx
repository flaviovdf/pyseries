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

from ..exceptions import DataFormatException

cdef inline int TRUE = 1
cdef inline int FALSE = 0

cdef int check_unique_sorted(double[:] array) nogil:
    '''Checks if the given array is sorted'''

    cdef Py_ssize_t i
    for i from 1 <= i < array.shape[0]:
        if array[i] <= array[i - 1]:
            return FALSE

    return TRUE

cdef int bin_search_pos(double[:] array, double value) nogil:
    '''
    Finds the first element in the array where the given is OR should have been
    in the given array. This is simply a binary search, but if the element is
    not found we return the index where it should have been at.

    This method is mainly used to filter dates from a time series.
    '''

    cdef Py_ssize_t n = array.shape[0]
    cdef Py_ssize_t lower = 0
    cdef Py_ssize_t upper = n - 1 #closed interval
    
    cdef Py_ssize_t half
    cdef Py_ssize_t idx = -1 
    while upper >= lower:
        half = lower + ((lower - upper) // 2)
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
        self.data = data
        self.timestamps = timestamps
        
        if check_unique_sorted(timestamps) == FALSE:
            raise DataFormatException('Timestamps must be sorted and unique')

    cdef TimeSeries filter_upper(self, double timestamp):
        '''
        Creates a new TimeSeries object with the elements which exist from 
        (open interval) a given date.
        
        Parameters
        ----------
        timestamp : double
            Date to filter
        '''
        cdef Py_ssize_t idx = bin_search_pos(self.data, timestamp)
        return TimeSeries(self.data[idx:], self.timestamps[idx:])

    cdef TimeSeries filter_lower(self, double timestamp):
        '''
        Creates a new TimeSeries object with the elements which exist up to
        (open interval) a given date.
        
        Parameters
        ----------
        timestamp : double
            Date to filter
        '''
        cdef Py_ssize_t idx = bin_search_pos(self.data, timestamp)
        return TimeSeries(self.data[:idx], self.timestamps[:idx])

    cdef TimeSeries filter_mid(self, double lowerstamp, double upperstamp):
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

cdef class TimeSeriesDataset(object):
    '''
    Represents a dataset with multiple time series. Each individual time series
    must be added to the dataset.
    '''
    def __init__(self, TimeSeries[:] series):
        self.series = series
        self.num_series = series.shape[0]

    def __getitem__(self, Py_ssize_t idx):
        return self.series[idx]
    
    cdef double[::1] to_numpy_like_data(self, str heuristic):

        return None
