# -*- coding: utf8
from __future__ import division, print_function

import numpy as np
cimport numpy as np

np.import_array()

cdef double PI = np.pi

cdef extern from 'math.h':
    double sin(double)

cdef double _decay(double beta, double tau, double n):
    return beta * (n ** tau)

cdef double _period(double pa, double ps, double pf, double n):
    return 1 - 0.5 * pa * (sin((PI * 2 / pf) * (ps + n)) + 1)

cdef double[:] _spikem(double[:] parameters, Py_ssize_t duration):
    
    cdef double pop = parameters[0]
    cdef double beta = 0
    if pop != 0:
        beta = parameters[1] / pop

    cdef double tau = -np.abs(parameters[2])
    cdef int nb = <int> np.round(parameters[3])
    cdef double shock = parameters[4]
    cdef double err = parameters[5]
    cdef double pa = parameters[6]
    cdef double ps = parameters[7]
    cdef double pf = parameters[8]
    
    cdef double[:] uninformed = np.zeros(duration, dtype='d')
    cdef double[:] delta_informed = np.zeros(duration, dtype='d')
    
    #Initialize model
    uninformed[0] = pop
    
    #RNF model
    cdef Py_ssize_t n = 0
    cdef Py_ssize_t t = 0
    cdef double sum_ = 0
    cdef double curr_shock = 0
    for n in range(duration - 1):

        sum_ = 0
        for t in range(nb, n + 1):
            if t == nb:
                curr_shock = shock
            else:
                curr_shock = 0

            sum_ += (delta_informed[t] + curr_shock) * \
                    _decay(beta, tau, n+1 - t)
        
        delta_informed[n + 1] = _period(pa, ps, pf, n) * \
                (uninformed[n] * sum_ + err)

        if delta_informed[n + 1] > uninformed[n]:
            #Hackish fix so that we cannot infect more than previous step.
            delta_informed[n + 1] = uninformed[n]
        uninformed[n + 1] = uninformed[n] - delta_informed[n + 1]
        
    return delta_informed

def decay(double beta, double tau, double n):
    '''
    Computes the shock function
    '''
    return _decay(beta, tau, n)

def period(double pa, double ps, double pf, double n):
    '''
    Computes the period function.
    '''
    return _period(pa, ps, pf, n)

def spikem_wrapper(double param_value, Py_ssize_t param_to_optimize,
        double[:] parameter_array, Py_ssize_t duration):
    '''
    Wrapper to call the spikem model varying one parameter at a time.
    '''
    parameter_array[param_to_optimize] = param_value
    return np.asarray(_spikem(parameter_array, duration))

def spikem(double[:] parameters, Py_ssize_t duration):
    '''
    Calls the SpikeM models

    Parameters
    ----------
    parameters : array like of length 8
        each one of the SpikeM parameters
    duration : int
        the length of the duration

    Returns
    -------
    a time series like array
    '''

    return np.asarray(_spikem(parameters, duration))
