# -*- coding: utf8
from __future__ import division, print_function

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from pyseries.data import tsio
from pyseries.prediction.temporal_weight import TemporalWeight
from pyseries.testing import YOUTUBE_1K

import numpy as np

def test_avg():
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2)

    tw = TemporalWeight('avg')
    y = tw.fit_predict(D)
    assert_array_equal(D.np_like_firstn().mean(axis=1), y)

def test_pow():
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()

    tw = TemporalWeight('pow', 2)
    y = tw.fit_predict(D)
    
    for i in xrange(D.shape[0]):
        div = sum((np.arange(D.shape[1]) + 1) ** 2)
        val = sum(D[i] * ((np.arange(D.shape[1]) + 1) ** 2) / div)
        
        assert_equal(val, y[i])

def test_yes():
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()

    tw = TemporalWeight('yes', 2)
    y = tw.fit_predict(D)
    
    for i in xrange(D.shape[0]):
        val = D[i][-1]
        assert_equal(val, y[i])