# -*- coding: utf8
from __future__ import division, print_function
'''
Tests for the io module
'''

from numpy.testing import assert_equal
from pyseries.testing import YOUTUBE_1K

from pyseries.data import tsio

def test_from_mat():
    dataset = tsio.from_id_row_mat(YOUTUBE_1K)
    assert_equal(1000, dataset.num_series)
    
    for series in dataset:
        assert_equal(100, len(series))