# -*- coding: utf8
from __future__ import division, print_function

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from pyseries.data import tsio
from pyseries.prediction.linear_model import RidgeRBFModel, MLModel
from pyseries.prediction.linear_model import mrse_transform
from pyseries.testing import YOUTUBE_1K

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
    rbf_model = RidgeRBFModel(10, .5, .01)
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-6).np_like_firstn()
    
    X_train = D[:500, :7]
    X_test = D[500:, :7]
    
    y_train = D.sum(axis=1)[:500]
    y_test = D.sum(axis=1)[500:]
    
    model = rbf_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mrse = (((y_test - y_pred) / y_test)**2).mean()
    assert_equal(1, mrse > 0)
    assert_equal(1, mrse < 1)

def test_rbf_with_dataset():
    '''Tests the RBF model with TimeSeriesDataset'''
    rbf_model = RidgeRBFModel(10, .5, .01)
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-6)
    y = D.np_like_firstn().sum(axis=1)
    
    model = rbf_model.fit(D, y)
    y_pred = model.predict(D)
    mrse = (((y - y_pred) / y)**2).mean()
    assert_almost_equal(0, mrse, 5)
    
def test_ml_model():
    '''Tests the ML model'''
    ml_model = MLModel()
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-6).np_like_firstn()
    D += 1e-6
    
    X_train = D[:500, :7]
    X_test = D[500:, :7]
    
    y_train = D.sum(axis=1)[:500]
    y_test = D.sum(axis=1)[500:]
    
    model = ml_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mrse = (((y_test - y_pred) / y_test)**2).mean()
    assert_equal(1, mrse > 0)
    assert_equal(1, mrse < 1)