# -*- coding: utf8
from __future__ import division, print_function

from pyseries.prediction.linear_model import mrse_transform

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

import numpy as np

def test_mrse_transform():
    '''Tests the mrse transformation'''
    
    X = np.asarray([[10.0, 20, 30], [5, 14, 18]])
    y = np.asarray([2.0, 10])
    
    Xt, yt = mrse_transform(X, y)
    
    assert_array_almost_equal(yt, np.ones(2))
    assert_array_almost_equal(Xt, [[5, 10, 15], [.5, 1.4, 1.8]])

def test_rbf_model():
    '''Tests the RBF model'''

    pass
