# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Module which contains functions to calculate entropy related metrics'''
from __future__ import division, print_function

cdef double INF = float('inf')

#Log2 from C99
cdef extern from "math.h" nogil:
    double log2(double)

cpdef double entropy(double[:] probabilities_x) nogil:
    '''
    Calculates the entropy (H) of the input vector which
    represents some random variable X.

    Parameters
    ----------
    probabilities_x : array like of doubles
        Array with the individual probabilities_x. Values must be 0 <= x <=1 
        and sum up to one
    '''
    cdef double return_val = 0
    cdef Py_ssize_t i = 0
    
    for i in range(probabilities_x.shape[0]):
        if probabilities_x[i] > 0:
            return_val -= probabilities_x[i] * log2(probabilities_x[i])
    
    return return_val

cpdef double mutual_information(double[:] probabilities_x, \
        double[:] probabilities_xy) nogil:
    '''
    Calculates the mutual information between the
    random variables (X and Y):

    Parameters
    ----------
    probabilities_x : array like of doubles
        Array with the individual probabilities X. Values must be 0 <= x <= 1
        and sum up to one

    probabilities_xy: array like of doubles
        Array with the individual probabilities for X|Y. Values must be 
        0 <= x <= 1 and sum up to one
    '''

    cdef double h_x = entropy(probabilities_x)
    cdef double h_xy = entropy(probabilities_xy)
    return h_x - h_xy

cpdef double norm_mutual_information(double[:] probabilities_x, \
        double[:] probabilities_xy) nogil:
    '''
    Calculates the normalized mutual information between the
    random variables (X and X|Y):

    Parameters
    ----------
    probabilities_x : numpy array or any iterable
        Array with the individual probabilities X. Values must be 0 <= x <= 1
        and sum up to one

    probabilities_xy : numpy array or any iterable
        Array with the individual probabilities for X|Y. 
        Values must be 0 <= x <= 1 and sum up to one
    '''

    cdef double h_x = entropy(probabilities_x)
    cdef double h_xy = entropy(probabilities_xy)

    cdef double normalized_mi = 0
    if h_x > 0 and h_xy > 0:
        normalized_mi = 1 - (h_x - h_xy) / h_x
        
    return normalized_mi

cpdef double kullback_leiber_divergence(double[:] probabilities_p, \
        double[:] probabilities_q) nogil:
    '''
    Calculates the Kullback-Leiber divergence between the distributions
    of two random variables.

    $$ D_{kl}(P(X) || Q(X)) = \sum_{x \in X) p(x) * log(\frac{p(x)}{q(x)}) $$

    Parameters
    ----------
    probabilities_p : array like of double
        Array with the individual probabilities P. Values must be 0 <= x <= 1
        and sum up to one

    probabilities_q : array like of double
        Array with the individual probabilities for Q. Values must be 0 <= x <= 1
        and sum up to one
    '''
    if probabilities_p.shape[0] != probabilities_q.shape[0]:
        return INF

    cdef double return_val = 0
    cdef double prob_p = 0
    cdef double prob_q = 0
    cdef Py_ssize_t i = 0

    for i in range(probabilities_p.shape[0]):
        prob_p = probabilities_p[i] 
        prob_q = probabilities_q[i]

        if prob_p != 0 and prob_q == 0:
            return INF
        elif prob_p > 0 and prob_q > 0:
            return_val += prob_p * (log2(prob_p) - log2(prob_q))
    return return_val
