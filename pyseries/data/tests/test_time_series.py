# -*- coding: utf8
from __future__ import division, print_function
'''
Tests for the base datasets module.
'''

from pyseries.data.base import TimeSeries

from pyseries.exceptions import DataFormatException
from pyseries.exceptions import ParameterException

from nose.tools import raises

from numpy.testing import assert_array_equal

import numpy as np

def test_timeseries_creation_ok():
    '''Valid creation test'''

    idx = np.array([0, 1, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    
    #OK creation
    ts = TimeSeries(idx, data)

@raises(DataFormatException)
def test_timeseries_creation_not_ok_repeated():
    '''Error because not unique stamps'''

    idx = np.array([0, 1, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')

    idx[0] = idx[1]
    ts = TimeSeries(idx, data)

@raises(DataFormatException)
def test_timeseries_creation_not_ok_inverted():
    '''Error because of inverted idx'''

    idx = np.array([0, 1, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')

    idx = idx[::-1]
    ts = TimeSeries(idx, data)

@raises(ParameterException)
def test_timeseries_creation_not_ok_shape():
    '''Error because of different shapes'''

    idx = np.array([1.0, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')

    ts = TimeSeries(idx, data)

def test_filter_lower():
    '''Tests the lower filter'''

    idx = np.array([0.0, 1.0, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    ts = TimeSeries(idx, data)

    result = ts.filter_lower(1)
    assert_array_equal(result.timestamps, [0.0])
    assert_array_equal(result.data, [1.0])

    result = ts.filter_lower(0)
    assert_array_equal(result.timestamps, [])
    assert_array_equal(result.data, [])

    result = ts.filter_lower(3)
    assert_array_equal(result.timestamps, [0.0, 1.0, 2.0])
    assert_array_equal(result.data, [1.0, 2.0, 3.0])

    result = ts.filter_lower(-1)
    assert_array_equal(result.timestamps, [])
    assert_array_equal(result.data, [])

    result = ts.filter_lower(8)
    assert_array_equal(result.data, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert_array_equal(result.timestamps, [0.0, 1.0, 2.0, 3.0, 4.0])

    expected = []
    for i, f in enumerate(np.arange(-0.5, 5.51, 1)):
        result = ts.filter_lower(f)
        assert_array_equal(result.timestamps, expected)
        if i < ts.timestamps.shape[0]:
            expected.append(ts.timestamps[i])

def test_filter_upper():
    '''Tests the upper filter'''

    idx = np.array([0.0, 1.0, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    ts = TimeSeries(idx, data)

    result = ts.filter_upper(1)
    assert_array_equal(result.timestamps, [1.0, 2.0, 3.0, 4.0])
    assert_array_equal(result.data, [2.0, 3.0, 4.0, 5.0])

    result = ts.filter_upper(2.5)
    assert_array_equal(result.timestamps, [3.0, 4.0])
    assert_array_equal(result.data, [4.0, 5.0])

    result = ts.filter_upper(-1)
    assert_array_equal(result.timestamps, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert_array_equal(result.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    result = ts.filter_upper(8)
    assert_array_equal(result.timestamps, [])
    assert_array_equal(result.data, [])

def test_filter_mid():
    '''Filters of a middle section'''

    idx = np.array([0.0, 1.0, 2, 3, 4], dtype='d')
    data = np.array([1.0, 2, 3, 4, 5], dtype='d')
    ts = TimeSeries(idx, data)

    result = ts.filter_mid(1, 3)
    assert_array_equal(result.timestamps, [1.0, 2.0])
    assert_array_equal(result.data, [2.0, 3.0])

    result = ts.filter_mid(0.5, 2.5)
    assert_array_equal(result.timestamps, [1.0, 2])
    assert_array_equal(result.data, [2.0, 3])

    result = ts.filter_mid(1.5, 3.5)
    assert_array_equal(result.timestamps, [2.0, 3.0])
    assert_array_equal(result.data, [3.0, 4.0])
