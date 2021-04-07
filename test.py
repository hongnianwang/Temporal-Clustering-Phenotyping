import tensorflow as tf
from data_loader import import_data
from utils import predictive_clustering_loss
from base import ACTPC

# Import data as tensorflow data Dataset
X_data, y_data = import_data(
    folder_path = 'data/sample/',
    data_name   = 'X'
)

init_model  = ACTPC(
    num_clusters = 12,
    latent_dim = 32,
    output_dim = y_data.shape[-1],
    alpha = 0.1)

init_model.initialise(
    X = X_data,
    y = y_data
)



