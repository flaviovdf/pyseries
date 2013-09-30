# -*- coding: utf8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division, print_function

from cpython cimport bool

import numpy as np
cimport numpy as np

def estimate_period(x, threshold):
    
    x = np.asanyarray(x)
    n = x.shape[0]
    
    mean = x.mean()
    cov = np.correlate(x - mean, x - mean, 'full') / n
    acf = cov[n - 1:]
    acf /= acf[0]
    acf = acf[1:] #first position is always 1, ignore
    
    argmx = acf.argmax()
    period = 0
    if acf[argmx] > threshold:
        period = argmx + 1

    return period

cdef class Model(object):
    
    def __init__(self):
        self.min_err = np.inf
        self.alpha = 0
        self.beta = 0
        self.d = 0
        self.m = 0
        self.gamma = 0
        self.y = None
        self.b = None
        self.l = None
        self.s = None
    
    def __call__(self, double[:] params, double[:] x, bool trend=False, 
            bool period=False, double period_threshold=0.5):
           
        cdef int m = 0
        if period:
            m = <int> estimate_period(x, period_threshold)
        
        cdef double alpha = params[0]
        cdef double beta = 0
        cdef double d = 0
        cdef double gamma = 0
    
        if trend:
            beta = params[1]
            d = params[2]
    
        if period:
            gamma = params[3]
        
        cdef double std = params[4]
        
        cdef Py_ssize_t n = x.shape[0]

        cdef double[:] y = np.zeros(n, dtype='d')
        cdef double[:] b = np.zeros(n, dtype='d')
        cdef double[:] l = np.zeros(n, dtype='d')
        cdef double[:] s = np.zeros(n, dtype='d')
        cdef double[:] noise 
        
        if std > 0:
            noise = np.random.normal(0.0, std, n)
        else:
            noise = np.zeros(n)
        
        y[0] = x[0]
        l[0] = x[0]
    
        if trend:
            b[0] = x[0]
    
        if period:
            s[0] = x[0]
            
        cdef Py_ssize_t i
        cdef double ssq_err = 0
        for i in range(1, n):
            if i >= m and m > 0:
                y[i] = l[i - 1] + d * b[i - 1] + s[i - m] + noise[i]
                s[i] = s[i - m] + gamma * noise[i]
            else:
                y[i] = l[i - 1] + d * b[i - 1] + s[i - 1] + noise[i]
                s[i] = s[i - 1]
                    
            l[i] = l[i - 1] + b[i - 1] + alpha * noise[i]
            b[i] = b[i - 1] + beta * noise[i]
        
            ssq_err += (x[i] - y[i]) * (x[i] - y[i])
        
        if ssq_err < self.min_err:
            self.min_err = np.inf
            self.alpha = alpha
            self.beta = beta
            self.d = d
            self.gamma = gamma
            self.std = std
            self.m = m
            self.y = y
            self.b = b
            self.l = l
            self.s = s
        
        return ssq_err
    
    def walk(self, int n_steps):
        
        cdef Py_ssize_t n = self.y.shape[0]
        
        cdef double[:] y = np.zeros(n_steps + 1, dtype='d')
        cdef double[:] b = np.zeros(n_steps + 1, dtype='d')
        cdef double[:] l = np.zeros(n_steps + 1, dtype='d')
        cdef double[:] s = np.zeros(n_steps + 1, dtype='d')
        
        cdef double alpha = self.alpha
        cdef double beta = self.beta
        cdef double d = self.d
        cdef double gamma = self.gamma
        cdef double std = self.std
        cdef int m = self.m
        
        y[0] = self.y[n - 1]
        l[0] = self.l[n - 1]
        b[0] = self.b[n - 1]
        s[0] = self.s[n - 1]
        
        cdef double[:] noise
        
        if std > 0:
            noise = np.random.normal(0.0, std, n_steps + 1)
        else:
            noise = np.zeros(n_steps + 1)
            
        cdef Py_ssize_t i
        for i in range(1, n_steps + 1):
            if i >= m and m > 0:
                y[i] = l[i - 1] + d * b[i - 1] + s[i - m] + noise[i]
                s[i] = s[i - m] + gamma * noise[i]
            else:
                y[i] = l[i - 1] + d * b[i - 1] + s[i - 1] + noise[i]
                s[i] = s[i - 1]
                    
            l[i] = l[i - 1] + b[i - 1] + alpha * noise[i]
            b[i] = b[i - 1] + beta * noise[i]
        
        return y[1:]