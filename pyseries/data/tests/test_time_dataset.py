# -*- coding: utf8
from __future__ import division, print_function
'''
Tests for the TimeSeriesDataset class.
'''

from pyseries.data.base import TimeSeries
from pyseries.data.base import TimeSeriesDataset

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

def test_to_numpy():
    '''Valid creation test'''

    idx1 = np.array([0, 1, 2, 3, 4], dtype='d')
    data1 = np.array([1.0, 2, 3, 4, 5], dtype='d')
    
    ts1 = TimeSeries(idx1, data1)

    idx2 = np.array([0, 1, 2, 3, 4, 5], dtype='d')
    data2 = np.array([1.0, 2, 3, 4, 8, 6], dtype='d')

    ts2 = TimeSeries(idx2, data2)

    dataset = TimeSeriesDataset(np.asarray([ts1, ts2]))

    X = dataset.np_like_firstn()
    assert_array_equal(X, [[1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 8]])
    
    X = dataset.np_like_lastn()
    assert_array_equal(X, [[1.0, 2, 3, 4, 5], [2, 3, 4, 8, 6]])

def test_iter():
    '''Iterates over the time series'''

    idx1 = np.array([0, 1, 2, 3, 4], dtype='d')
    data1 = np.array([1.0, 2, 3, 4, 5], dtype='d')
    
    ts1 = TimeSeries(idx1, data1)

    idx2 = np.array([0, 1, 2, 3, 4, 5], dtype='d')
    data2 = np.array([1.0, 2, 3, 4, 8, 6], dtype='d')

    ts2 = TimeSeries(idx2, data2)

    dataset = TimeSeriesDataset(np.asarray([ts1, ts2]))
    for i, timeseries in enumerate(dataset):
        if i == 0:
            assert_equal(5, len(timeseries))
        if i == 1:
            assert_equal(6, len(timeseries))

        if i > 1:
            raise Exception()

def test_to_empty():
    '''Valid empty creation test'''

    idx1 = np.array([], dtype='d')
    data1 = np.array([], dtype='d')
    
    ts1 = TimeSeries(idx1, data1)
    ts2 = TimeSeries(idx1.copy(), data1.copy())

    dataset = TimeSeriesDataset(np.asarray([ts1, ts2]))

    X = dataset.np_like_firstn()
    assert_array_equal(X, [[], []])
    
    X = dataset.np_like_lastn()
    assert_array_equal(X, [[], []])
