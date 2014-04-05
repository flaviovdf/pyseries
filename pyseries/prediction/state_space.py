# -*- coding: utf8
from __future__ import division, print_function
'''
Implements models based on state space models.
'''

from _ssm_model import Model
from pyseries.data.base import TimeSeriesDataset

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

import numpy as np

class SSM(BaseEstimator, RegressorMixin):
    
    def __init__(self, trend=False, period=False, steps_ahead=1, n_walks=5,
                 normalize_err=False):
        self.P = None
        self.T = None
        self.trend = trend
        self.period = period
        self.steps_ahead = steps_ahead
        self.n_walks = n_walks
        self.normalize_err = normalize_err
    
    def fit_predict(self, X, full_series=False):

        if not isinstance(X, TimeSeriesDataset):
            X = np.asanyarray(X, dtype='d')
        else:
            X = X.np_like_firstn()
        
        num_params = 5
        init_params = np.random.rand(num_params)
        init_params[-1] = 1e-20
        n, ticks = X.shape

        if not full_series:
            Y = np.zeros(shape=(n, self.steps_ahead), dtype='d')
        else:
            Y = np.zeros(shape=(n, ticks + self.steps_ahead), dtype='d')
        
        bounds = [(None, None), (None, None), (None, None), (None, None), 
                (0, None)]
        for i in xrange(X.shape[0]):
            model = Model(self.normalize_err)
            fmin_l_bfgs_b(func=model, x0=init_params,
                    approx_grad=True, bounds=bounds,
                    args=(X[i], self.trend, self.period),
                    iprint=-1)
            for _ in xrange(self.n_walks):
                y = np.asarray(model.walk(self.steps_ahead, full_series))
                Y[i] += y
        Y /= self.n_walks
        return Y
