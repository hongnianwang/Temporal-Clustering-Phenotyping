#!/usr/bin/env python3
"""
Functions used to define AC-TPC model.

Updated: 02 December 2020
Created by: Henrique Aguiar, Institute of Biomedical Engineering, Department of Engineering Science, 
University of Oxford

if you have any queries, please reach me at henrique.aguiar@ndcn.ox.ac.uk

This script implements AC-TPC utility functions in tensorflow 2.3.0. The original model is described in paper:
    
    "Temporal Phenotyping using Deep Predictive Clustering of Disease Progression", by C. Lee and M. van der Schaar.
    
A github implementation in tensorflow 1.0.6 can be found:
    
    https://github.com/chl8856/AC_TPC
    
ts = time-series (one, or perhaps a batch of them)
                
"""
import tensorflow as tf
from tensorflow import math


def log(x): 
    "Log function to avoid zeros"
    return tf.log(x + 1e-16)



def div(x, y):
    "Divisor function to avoid zeros"
    return tf.div(x, (y + 1e-8))



def get_ts_length_(ts):
    "Obtain length of each individual time-series."
    "Input is batch size x time-steps (variable) x number of dimensions"

    max_across_feat_ = (math.reduce_max(math.abs(ts), axis = 2))                       # vector of max absolute values over feature dimensions
    num_time_steps_  = math.reduce_sum(math.sign(max_across_feat_), axis = 1)          # number of time steps. sign(0) = 0, so dimensions without values don't add as time steps.
    
    ts_length_vec    =  tf.cast(num_time_steps_, dtype = tf.int32)
    
    return ts_length_vec


    
    























