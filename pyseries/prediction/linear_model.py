# -*- coding: utf8
from __future__ import division, print_function
'''
Implements models based on linear regression. In general these models are
only wrappers to sklearn models.
'''

import numpy as np

from ..exceptions import ParameterException
from ..exceptions import ShapeException

from scipy.spatial.distance import pdist

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from sklearn.linear_model import Ridge

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

    if y.ndim > 1:
        raise ShapeException('y must have a single dimension')

    if X.shape[0] != y.shape[0]:
        raise ShapeException('X and y must have the same number of rows!')

    X_new = (X.T / y).T
    y_new = np.ones(y.shape)

    return X_new, y_new

class RidgeRBFModel(BaseEstimator, RegressorMixin):
    
    '''
    Implements rbf model by Pinto et al. 2013.

    Parameters
    ----------
    num_dists : integer
        number of distances to consider
    sigma : float
        smoothing in the rbf
    alpha : float
        ridge regression scaling parameter
    '''

    def __init__(self, num_dists, sigma, alpha):
        self.num_dists = num_dists
        self.sigma = sigma
        self.alpha = alpha
        self.R = None
        self.model = None

    def fit(X, y):
        X = np.asanyarray(X, dtype='f')
        y = np.asanyarray(y, dtype='f')
        
        X, y = mrse_transform(X, y)

        n = X.shape[0]
        
        if self.num_dists > n:
            raise ParameterException('Number of distances is greater than ' + \
                    'num rows in X')

        if self.num_dists <= 0:
            self.R = None
        else:
            self.R = np.random.choice(X.shape[0], num_dists, replace=False)
            D = np.exp(-pdist(X, self.R) / (2 * (self.sigma ** 2)))
            X = np.hstack((X, D))
        
        self.model = Ridge(self.alpha, fit_intercept=False)
        self.model = self.model.fit(X, y)
        return self

    def predict(X):
        X = np.asanyarray(X, dtype='f')

        if self.R is not None:
            D = np.exp(-pdist(X, self.R) / (2 * (self.sigma ** 2)))
            X = np.hstack((X, D))

        return self.model.predict(X)

class MLModel(RidgeRBFModel):
    '''
    Implements the MLModel by Pinto et al. 2013.
    '''

    def __init__(self):
        super(RidgeRBFModel, self).__init__(0, 0, 1)
