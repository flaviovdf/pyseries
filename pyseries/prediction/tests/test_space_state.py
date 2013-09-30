# -*- coding: utf8
from __future__ import division, print_function

from numpy.testing import assert_equal

from pyseries.data import tsio
from pyseries.prediction.state_space import SSM
from pyseries.testing import YOUTUBE_1K

def test_fit_predict():
    '''
    Tests the fit and predict methods for the state space models. Trend + period
    '''
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()
    D = D[: 50]
    ssm = SSM(True, True, 50)
    
    X_train = D[:, :50]
    X_test = D[:, 50:]
    
    Y = ssm.fit_predict(X_train)
    err = ((X_test - Y) ** 2).mean()
    assert_equal((50, 50), Y.shape)
    assert err > 0


def test_fit_predict1():
    '''
    Tests the fit and predict methods for the state space models. Trend only
    '''
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()
    D = D[: 50]
    ssm = SSM(True, False, 50)
    
    X_train = D[:, :50]
    X_test = D[:, 50:]
    
    Y = ssm.fit_predict(X_train)
    err = ((X_test - Y) ** 2).mean()
    assert_equal((50, 50), Y.shape)
    assert err > 0
    
def test_fit_predict2():
    '''
    Tests the fit and predict methods for the state space models. Period Only
    '''
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()
    D = D[: 50]
    ssm = SSM(False, True, 50)
    
    X_train = D[:, :50]
    X_test = D[:, 50:]
    
    Y = ssm.fit_predict(X_train)
    err = ((X_test - Y) ** 2).mean()
    assert_equal((50, 50), Y.shape)
    assert err > 0


def test_fit_predict3():
    
    '''
    Tests the fit and predict methods for the state space models. Smooth Only
    '''
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()
    D = D[: 50]
    ssm = SSM(False, False, 50)
    
    X_train = D[:, :50]
    X_test = D[:, 50:]
    
    Y = ssm.fit_predict(X_train)
    err = ((X_test - Y) ** 2).mean()
    assert_equal((50, 50), Y.shape)
    assert err > 0

def test_fit_predict4():
    
    '''
    Tests the fit and predict methods for the state space models. Smooth Only
    '''
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()
    D = D[: 50]
    ssm = SSM(False, False, 50)
    
    X_train = D[:, :50]
    X_test = D[:, 50:]
    
    Y = ssm.fit_predict(X_train)
    err = ((X_test - Y) ** 2).mean()
    assert_equal((50, 50), Y.shape)
    assert err > 0

def test_fit_predict_one_step():
    
    '''
    Tests the fit and predict methods for the state space models. Smooth Only
    '''
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2).np_like_firstn()
    D = D[: 50]
    ssm = SSM(False, False, 1)
    
    X_train = D[:, :50]
    X_test = D[:, 50:]
    
    Y = ssm.fit_predict(X_train)
    err = ((X_test - Y) ** 2).mean()
    assert_equal((50, 1), Y.shape)
    assert err > 0
    
def test_fit_predict_dataset():
    '''Tests with dataset'''
    
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2)
    ssm = SSM(False, False, 1)
    Y = ssm.fit_predict(D)
    assert Y.any()
    
def test_fit_predict_normalize():
    '''Tests with dataset'''
    
    D = tsio.from_id_row_mat(YOUTUBE_1K, add_eps=1e-2)
    ssm = SSM(False, False, 1, normalize_err=True)
    Y = ssm.fit_predict(D)
    assert Y.any()
