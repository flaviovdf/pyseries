# -*- coding: utf8
from __future__ import division, print_function
'''
Implements models based on the SpikeM paper.
'''

from _spikem_model import spikem
from _spikem_model import spikem_wrapper

from pyseries.data.base import TimeSeriesDataset

from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import leastsq

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

import lmfit
import numpy as np

PARAMETERS = {
        'N':0,
        'beta':1,
        'tau':2,
        'nb':3,
        'shock':4,
        'err':5,
        'pa':6,
        'ps':7,
        'pf':8}

INV_PARAMETERS = dict((v, k) for k, v in PARAMETERS.items())

TOTAL_PARAMS = len(PARAMETERS)

TO_FIT = ['nb', 'shock', 'beta', 'N', 'ps', 'pa', 'err']

def get_initial_parameters(tseries, period):
    '''
    Initates parameters estimates with 'good guesses' for this time series

    Parameters
    ----------
    tseries : numpy array like
        time series
    period: int
        the period of the time series

    Returns
    -------
    initial parameters in the form of an array and the bounds
    '''

    init_params = np.zeros(TOTAL_PARAMS, dtype='d')
    bounds = []

    #N
    init_params[0] = tseries.sum()
    bounds.append((tseries.sum(), None))
    
    #beta
    init_params[1] = 1.0
    bounds.append((0, 2))
    
    #tau
    init_params[2] = -1.5
    bounds.append((-1.5, -1.5))

    #nb
    init_params[3] = 0
    bounds.append((0, tseries.shape[0]))

    #Sb
    init_params[4] = 0.1
    bounds.append((0, None))

    #err
    init_params[5] = 0.01
    bounds.append((0, 0.1))
    
    #pa - rate
    init_params[6] = 0.01
    bounds.append((0, 1))

    #ps - shift
    init_params[7] = 0.01
    bounds.append((0.05, period))
    
    #pf - period
    init_params[8] = period
    bounds.append((period, period))

    return init_params, bounds
    

def to_lmstyle(curr_vary, curr_fit, bounds):

    params_lmfit = lmfit.Parameters()
    for p in PARAMETERS:
        init = curr_fit[PARAMETERS[p]]
        bound = bounds[PARAMETERS[p]]
        
        vary = p == curr_vary
        #bound[1] > bound[0]
        params_lmfit.add(p, value=init, vary=vary, min=bound[0], max=bound[1])
    return params_lmfit

def fit_one(tseries, period, num_iter=20):
    duration = tseries.shape[0]

    init_params, bounds = get_initial_parameters(tseries, period)
    curr_params = init_params.copy()
    
    nonz = np.where(tseries > 0)[0]
    if nonz.shape[0] > 0:
        st = max(nonz[0] - 1, 0)
    else:
        st = 0
    
    ed = tseries.argmax()

    def mse_func(lm_params, param_name, all_parameters):
        param_value = lm_params[param_name].value
        parameter_idx = PARAMETERS[param_name]
        predicted = spikem_wrapper(param_value, parameter_idx, all_parameters,
                tseries.shape[0])
        
        return np.array([np.sqrt(((predicted - tseries) ** 2).mean())])

    for _ in range(num_iter):
        for param_name in TO_FIT:
            param_idx = PARAMETERS[param_name]
        
            if param_name == 'nb':
                best = 0
                min_mse = float('inf')
                
                copy_params = curr_params.copy()
                for i in xrange(st, ed):
                    copy_params[param_idx] = i
                    predicted = spikem(copy_params, tseries.shape[0])
                    
                    try:
                        mse = np.sqrt(((predicted - tseries) ** 2).mean())
                    except FloatingPointError: #overflow
                        mse = float('inf')

                    if mse < min_mse:
                        best = i
                
                curr_params[param_idx] = best
            else:
                bound = bounds[param_idx]
                lm_params = to_lmstyle(param_name, curr_params, bounds)
                lmfit.minimize(mse_func, lm_params, \
                        args=(param_name, curr_params))
                curr_params[param_idx] = lm_params[param_name].value
    
    return curr_params
 
class SpikeM(BaseEstimator, RegressorMixin):
    
    def __init__(self, steps_ahead=1):
        self.steps_ahead = steps_ahead

    def fit_predict(self, X, period_frequencies=None, full_series=False, num_iter=20):

        if not isinstance(X, TimeSeriesDataset):
            X = np.asanyarray(X, dtype='d')
        else:
            X = X.np_like_firstn()

        n = X.shape[0]
        if period_frequencies is None:
            period_frequencies = np.ones(n)
        else:
            period_frequencies = np.asanyarray(period_frequencies, dtype='d')

        assert period_frequencies.shape[0] == n
        
        P = np.zeros(shape=(n, TOTAL_PARAMS), dtype='d')
        with np.errstate(over='raise'):
            for i in xrange(X.shape[0]):
                P[i] = fit_one(X[i], period_frequencies[i], num_iter)
        
        if full_series:
            Y = np.zeros((n, X.shape[1] + self.steps_ahead), dtype='d')
        else:
            Y = np.zeros((n, ), dtype='d')

        for i in xrange(X.shape[0]):
            if full_series:
                Y[i] = spikem(P[i], X.shape[1] + self.steps_ahead)
            else:
                Y[i] = spikem(P[i], X.shape[1] + self.steps_ahead)[-1]
        return Y
