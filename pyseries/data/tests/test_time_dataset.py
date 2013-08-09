# -*- coding: utf8
from __future__ import division, print_function
'''
Tests for the TimeSeriesDataset class.
'''

from pyseries.data.base import TimeSeries
from pyseries.data.base import TimeSeriesDataset

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
