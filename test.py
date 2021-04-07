import os
import numpy as np
import tensorflow as tf
from data_loader import import_data
from utils import predictive_clustering_loss
from base import ACTPC

# Import data as tensorflow data Dataset
X_data, y_data = import_data(
    folder_path = 'data/sample/',
    data_name   = 'X'
)

# Convert to Tensor Dataset
X = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X_data, dtype='float32'))
y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_data, dtype='float32'))

init_model  = ACTPC(
    num_clusters = 12,
    output_dim = next(y.as_numpy_iterator()).shape[-1],
    alpha = 0.1)

init_model.initialise(
    X = X,
    y = y
)


