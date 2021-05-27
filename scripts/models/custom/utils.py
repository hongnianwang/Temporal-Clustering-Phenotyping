#!/usr/bin/env python3
"""
Auxiliary functions for base ACTPC.

At the moment, currently temporal loss implementations are computed.
"""
import numpy as np
from scipy.stats import mode

def compute_mode(array):
    
    "Compute mode for numpy array, across axis 1 (time-wise)"
    assert len(array.shape) == 3
    
    # Compute mode 
    array_mode = mode(array, axis = 1, nan_policy = "omit")
    
    return array_mode[0].reshape(-1, array.shape[-1])

























