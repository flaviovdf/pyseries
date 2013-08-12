# -*- coding: utf8

cpdef double entropy(double[:] probabilities_x) nogil

cpdef double mutual_information(double[:] probabilities_x, \
        double[:] probabilities_xy) nogil
                         
cpdef double norm_mutual_information(double[:] probabilities_x, \
        double[:] probabilities_xy) nogil
                              
cpdef double kullback_leiber_divergence(double[:] probabilities_p, 
        double[:] probabilities_q) nogil
