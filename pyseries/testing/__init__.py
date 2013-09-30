# -*- coding: utf8
from __future__ import print_function, division
'''
Testing utilities such as datasets
'''

import os

#I don't know if this the best way to locate test files. But it works.
DATA_DIR = os.path.join(__path__[0], 'data')
YOUTUBE_1K = os.path.join(DATA_DIR, 'tseries1k.dat')