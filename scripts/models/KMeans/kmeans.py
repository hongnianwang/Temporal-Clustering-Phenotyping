#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:45:45 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org

Script for running KMeans analysis.

Options available are:
    - Distance-sensitive or no (TSKM or Euclidean)
    - other params
"""
import numpy as np

import tslearn as tsl
from tslearn.clustering import TimeSeriesKMeans as TSkm 
from tslearn.clustering import KernelKMeans as Kkm 
from tslearn.clustering import KShape as KS

from sklearn.cluster import KMeans as km

def init_kmeans_model(algorithm, K, metric, verbose, init, seed, kernel = "gak", **kwargs):
    
    """
    Load model depending on algorithm choice
    """
    if algorithm == "TimeSeriesKMeans" or algorithm == "TSKM":
        model = TSkm(n_clusters = K, metric = metric, 
                     init = init, verbose = verbose, random_state = seed, **kwargs)
        
    elif algorithm == "KernelKMeans" or algorithm == "KKM":
        model = Kkm(n_clusters = K, kernel = kernel, 
                    verbose = verbose, random_state = seed, **kwargs)
        
    elif algorithm == "KShape" or algorithm == "KS":
        model = KS(n_clusters = K, 
                   verbose = verbose, random_state = seed, init = init, **kwargs)
        
    elif algorithm == "KMeans" or algorithm == "km":
        model = km(n_clusters = K, init = init,
                   verbose = verbose, random_state = seed, **kwargs)
    
    else:
        print("Wrong algorithm specified")
        return None
    
    return model




