# -*- coding: utf8
from __future__ import division, print_function
'''
Implements models based on linear regression. In general these models are
only wrappers to sklearn models.
'''

import numpy as np

from pyseries.data.base import TimeSeriesDataset

from pyseries.exceptions import ParameterException
from pyseries.exceptions import ShapeException

from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from sklearn.linear_model import LinearRegression
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
    X = np.asanyarray(X, dtype='d')
    y = np.asanyarray(y, dtype='d')

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
    '''

    def __init__(self, num_dists=2, sigma=0.1, **kwargs):
        self.num_dists = num_dists
        self.sigma = sigma
        self.base_kwags = kwargs
        self.R = None
        self.model = None

    def fit(self, X, y, base_learner=None):
        if isinstance(X, TimeSeriesDataset):
            X = X.np_like_firstn()

        if base_learner is None:
            base_learner = Ridge

        X = np.asanyarray(X, dtype='d')
        y = np.asanyarray(y, dtype='d')
        
        n = X.shape[0]
        num_dists = self.num_dists
        
        if self.num_dists > n:
            raise ParameterException('Number of distances is greater than ' + \
                    'num rows in X')

        if self.num_dists <= 0:
            self.R = None
        else:
            rand_idx = np.random.choice(X.shape[0], num_dists, replace=False)
            self.R = X[rand_idx]
            
            D = np.exp(-1.0 * ((cdist(X, self.R) ** 2) / (2 * (self.sigma ** 2))))
            X = np.hstack((X, D))

        X, y = mrse_transform(X, y)
        self.model = base_learner(**self.base_kwags)
        self.model = self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, TimeSeriesDataset):
            X = X.np_like_firstn()
             
        X = np.asanyarray(X, dtype='d')

        if self.R is not None:
            D = np.exp(-1.0 * ((cdist(X, self.R) ** 2) / (2 * (self.sigma ** 2))))
            X = np.hstack((X, D))

        return self.model.predict(X)

class MLModel(RidgeRBFModel):
    '''
    Implements the MLModel by Pinto et al. 2013.
    '''

    def __init__(self, **kwargs):
        super(MLModel, self).__init__(0, 0, **kwargs)

    def fit(self, X, y):
        super(MLModel, self).fit(X, y, LinearRegression)
        return self
