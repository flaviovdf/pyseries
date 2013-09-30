# -*- coding: utf8
from __future__ import division, print_function
'''
Code for loading time series from text files
'''

from pyseries.data.base import TimeSeries
from pyseries.data.base import TimeSeriesDataset

import numpy as np

def from_id_row_mat(matrix_fpath, skip_first_col=True, add_eps=0):
    '''
    Converts a file where each row composes of equally spaced time series
    observations. Moreover, all series have the same number of observations.
    The first row in the id of the time series, which will be ignored.
    
    Parameters
    ----------
    matrix_fpath : str 
        path to the matrix file
    skip_first_col : bool
        indicates that first column is an id which can be ignored
    add_eps : int (default=0)
        eps to add to each observatio
    
    Returns
    -------
    a time series dataset object
    '''
    
    from_ = int(skip_first_col)
    X = np.genfromtxt(matrix_fpath, dtype='d')[:, from_:] + add_eps
    
    n, f = X.shape
    tseries = []
    for i in xrange(n):
        idx = np.arange(f, dtype='d')
        data = X[i]
        
        ts = TimeSeries(idx, data)
        tseries.append(ts)
        
    return TimeSeriesDataset(np.asarray(tseries)) 