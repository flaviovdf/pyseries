# -*- coding: utf8
from __future__ import division, print_function
'''
Implements models based on state space models.
'''

from pyseries.data.base import TimeSeriesDataset

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

import numpy as np
from pyseries.exceptions import ParameterException

class TemporalWeight(BaseEstimator, RegressorMixin):
    
    def __init__(self, weight='avg', p=1):
        
        possible_weights = ['avg', 'pow', 'yes']
        if weight not in ['avg', 'pow', 'yes']:
            raise ParameterException('Param weight must be in ' + \
                                     possible_weights)
        
        self.weight = weight
        self.p = p
    
    def fit_predict(self, X):
        
        if isinstance(X, TimeSeriesDataset):
            X = X.np_like_firstn() 

        X = np.asanyarray(X, dtype='d')
        
        if self.weight == 'avg':
            return X.mean(axis=1)
        elif self.weight == 'pow':
            w = (np.arange(X.shape[1]) + 1) ** self.p
            w_merge = np.concatenate((w, ) * X.shape[0])
            X_new = (X.ravel() * w_merge).reshape(X.shape) / w.sum()
            return X_new.sum(axis=1)
        else:
            return X[:, -1]