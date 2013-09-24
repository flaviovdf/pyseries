#-*- coding: utf8

from pyseries.data.base cimport TimeSeries
from pyseries.data.base cimport TimeSeriesDataset

from pyseries.metrics cimport arr_entropy

cimport numpy as np

cpdef double entropy(TimeSeries timeseries):
    return arr_entropy.entropy(timeseries._data)

cpdef double entropy_rate_h_diff(TimeSeries timeseries):
    cdef Py_ssize_t n = timeseries._size
    return arr_entropy.entropy(timeseries._data[:n]) - \
            arr_entropy.entropy(timeseries._data[:n - 1])

cpdef double entropy_rate_h_lnorm(TimeSeries timeseries):
    cdef Py_ssize_t n = timeseries._size
    return arr_entropy.entropy(timeseries._data) / n

cpdef np.ndarray[double, ndim=1] dataset_entropy(TimeSeriesDataset dataset):
    return np.asarray([0.0])
