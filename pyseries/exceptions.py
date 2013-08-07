# -*- coding: utf8
from __future__ import division, print_function

'''
This module contains the core Exceptions thrown by the PySeries.
'''

class ShapeException(Exception):
    '''
    Thrown when numpy arrays have incompatible shapes for a given operation.
    '''
    pass
