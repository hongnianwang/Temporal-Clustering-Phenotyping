import os
import numpy as np
import tensorflow as tf
from data_loader import import_data
from utils import predictive_clustering_loss
from base import ACTPC

# Import data as tensorflow data Dataset
X, y = import_data(
    folder_path = 'data/sample/',
    data_name   = 'X'
)

init_model  = ACTPC(
    num_clusters = 12,
    output_dim = next(y.as_numpy_iterator()).shape[-1],
    alpha = 0.1)

init_model.init_train(
    X = X,
    y = y
)

