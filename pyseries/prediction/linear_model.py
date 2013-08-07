# -*- coding: utf8
from __future__ import division, print_function
'''
Implements models based on linear regression. In general these models are
only wrappers to sklearn models.
'''

import numpy as np

from ..exceptions import ShapeException
from sklearn import linear_model

def mrse_transform(X, y):
    '''
    Transforms X so that mean relative error regression models can fit. More
    formally this is making the indexes [i, j] of x equal to:: 
    
        X[i, j] / y[j]

    y is also converted to a ones array. Fitting linear regression models on
    such data will optimize for mean relative squared error.
    
    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        Input samples
    y : array of shape = [n_samples]
        Response variable

    Returns
    -------
    X_new : array-like of shape = [n_samples, n_features]
        Transformed X
    y_new : array of shape = [n_samples]
        Array of ones
    '''
    X = np.asanyarray(X, dtype='f')
    y = np.asanyarray(y, dtype='f')

    if X.shape[1] != y.shape[1]:
        raise ShapeException('X and y must have the same number of columns!')

    X_new = (X.T / y).T
    y_new = np.ones(y.shape)

    return X_new, y_new
