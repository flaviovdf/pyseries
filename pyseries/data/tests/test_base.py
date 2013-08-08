# -*- coding: utf8
from __future__ import division, print_function
'''
Tests for the base datasets module.
'''

from pyseries.data.base import TimeSeries
from pyseries.data.base import TimeSeriesDataset

from pyseries.exceptions import DataFormatException
from pyseries.exceptions import ParameterException

from nose.tools import raises

from numpy.testing import assert_array_equal

import numpy as np

def test_timeseries_creation_ok():
    '''Valid creation test'''

    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    idx = np.arange(data.shape[0], dtype='d')
    
    #OK creation
    ts = TimeSeries(data, idx)
    assert True

@raises(DataFormatException)
def test_timeseries_creation_not_ok_repeated():
    '''Error because not unique stamps'''

    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    idx = np.arange(data.shape[0], dtype='d')

    idx[0] = idx[1]
    ts = TimeSeries(data, idx)

@raises(DataFormatException)
def test_timeseries_creation_not_ok_inverted():
    '''Error because of inverted idx'''

    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    idx = np.arange(data.shape[0], dtype='d')

    idx = idx[::-1]
    ts = TimeSeries(data, idx)
    assert False

@raises(ParameterException)
def test_timeseries_creation_not_ok_shape():
    '''Error because of different shapes'''

    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    idx = np.arange(data.shape[0] - 1, dtype='d')

    ts = TimeSeries(data, idx)
    assert False

def test_filter_upper():
    '''Tests the upper filter'''

    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    idx = np.arange(data.shape[0], dtype='d')
    ts = TimeSeries(data, idx)

    result = ts.filter_lower(1)
    assert_array_equal(result.data, [1.0])
