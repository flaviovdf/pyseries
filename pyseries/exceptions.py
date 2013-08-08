# -*- coding: utf8
from __future__ import division, print_function

'''
This module contains the core Exceptions thrown by the PySeries.
'''

class DataFormatException(Exception):
    '''Base exception for any data related issues'''
    pass

class ShapeException(DataFormatException):
    '''
    Thrown when numpy arrays have incompatible shapes for a given operation.
    '''
    pass

class ParameterException(Exception):
    '''Indicates a wrong parameter for a given function'''
    pass
