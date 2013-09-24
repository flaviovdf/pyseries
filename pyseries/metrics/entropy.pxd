#-*- coding: utf8

from pyseries.data.base cimport TimeSeries
from pyseries.data.base cimport TimeSeriesDataset

from pyseries.metrics cimport arr_entropy

import numpy as np
cimport numpy as np

cpdef double entropy(TimeSeries timeseries)

cpdef double entropy_rate_h_diff(TimeSeries timeseries)

cpdef double entropy_rate_h_lnorm(TimeSeries timeseries)

cpdef np.ndarray[double, ndim=1] dataset_entropy(TimeSeriesDataset dataset)
